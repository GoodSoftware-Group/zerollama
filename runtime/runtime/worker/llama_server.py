"""Subprocess wrapper for llama-server (GGUF inference)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from runtime.logutil import get_logger
from runtime.worker.sampler_options import (
    SamplerOptions,
    apply_sampler_to_completion_payload,
)
from runtime.worker.llama_stream import iter_llama_sse_lines
from runtime.worker.llama_server_http import (
    cancellable_http_post,
    iter_response_chunks,
    read_response_body,
)

logger = get_logger("llama_server")

_GPU_BACKEND_PREFIXES = ("libggml-", "ggml-")


def _is_gpu_ggml_backend(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith((".so", ".dll", ".dylib")) and ".so." not in name:
        return False
    for prefix in _GPU_BACKEND_PREFIXES:
        if not name.startswith(prefix):
            continue
        if any(
            name.startswith(f"{prefix}{skip}")
            for skip in ("base", "cpu")
        ):
            return False
        return True
    return False


def _discover_gpu_backend_path() -> Path | None:
    """Match Go llama-server spawn: set GGML_BACKEND_PATH for CUDA offload."""
    explicit = os.environ.get("GGML_BACKEND_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    search_roots: list[Path] = []
    for key in ("OLLAMA_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        for part in raw.split(os.pathsep):
            if part:
                search_roots.append(Path(part))
    llama_bin = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if llama_bin:
        search_roots.insert(0, Path(llama_bin).resolve().parent)
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        matches = sorted(resolved.glob("libggml-*.so*"))
        for match in matches:
            if _is_gpu_ggml_backend(match):
                return match.resolve()
    return None


def _llama_server_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    backend = _discover_gpu_backend_path()
    if backend is not None:
        env["GGML_BACKEND_PATH"] = str(backend)
    return env


class LlamaServerError(Exception):
    pass


@dataclass
class LlamaServerProcess:
    """Manage a llama-server child process."""

    binary: Path
    model: Path
    host: str = "127.0.0.1"
    port: int = 8082
    n_gpu_layers: int = -1
    _proc: subprocess.Popen[bytes] | None = None
    _exit_code: int | None = field(default=None, init=False, repr=False)
    _crash_tail: str = field(default="", init=False, repr=False)
    _watchdog_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _watchdog_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _log_file: Any = field(default=None, init=False, repr=False)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status_snapshot(self) -> dict[str, Any]:
        """Operator snapshot for /health (no blocking I/O beyond poll)."""
        running = self.is_running()
        out: dict[str, Any] = {
            "running": running,
            "died": self._exit_code is not None and not running,
            "exit_code": self._exit_code,
            "reachable": None,
            "base_url": self.base_url,
        }
        if self._crash_tail:
            out["crash_tail"] = self._crash_tail[-2000:]
        if running:
            out["reachable"] = self._probe_health(timeout_s=1.0)
        return out

    def _probe_health(self, timeout_s: float = 2.0) -> bool:
        url = f"{self.base_url}/health"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _open_log_sink(self) -> int:
        log_path = os.environ.get("ZEROLLAMA_LLAMA_SERVER_LOG", "").strip()
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open("ab", buffering=0)
            return self._log_file.fileno()
        return subprocess.DEVNULL

    def _start_watchdog(self) -> None:
        self._watchdog_stop.clear()
        proc = self._proc
        if proc is None:
            return

        def _watch() -> None:
            assert proc is not None
            proc.wait()
            code = proc.returncode
            self._exit_code = code
            tail = ""
            if proc.stderr is not None:
                try:
                    raw = proc.stderr.read()
                    if raw:
                        tail = raw.decode(errors="replace")
                except OSError:
                    pass
            self._crash_tail = tail
            if code not in (0, None):
                logger.warning(
                    "llama-server exited model=%s code=%s tail=%s",
                    self.model.name,
                    code,
                    tail[-500:] if tail else "",
                )

        self._watchdog_thread = threading.Thread(
            target=_watch, name="llama-server-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, extra_args: list[str] | None = None) -> None:
        if self._proc is not None and self._proc.poll() is None:
            logger.info(
                "model %s already loaded (gguf=%s, url=%s)",
                self.model.name,
                self.model.resolve(),
                self.base_url,
            )
            return
        if not self.binary.is_file():
            raise LlamaServerError(f"llama-server not found: {self.binary}")
        if not self.model.is_file():
            raise LlamaServerError(f"model not found: {self.model}")

        model_path = str(self.model.resolve())
        logger.info(
            "loading model %s (gguf=%s, llama-server=%s, listen=%s:%d)",
            self.model.name,
            model_path,
            self.binary.name,
            self.host,
            self.port,
        )

        cmd = [
            str(self.binary),
            "--model",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ngl",
            str(self.n_gpu_layers),
        ]
        if extra_args:
            cmd.extend(extra_args)

        log_fd = self._open_log_sink()
        self._exit_code = None
        self._crash_tail = ""
        self._proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=log_fd if log_fd != subprocess.DEVNULL else subprocess.DEVNULL,
            env=_llama_server_subprocess_env(),
        )
        self._start_watchdog()
        self._wait_healthy(timeout_s=120.0)

    def _wait_healthy(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                tail = self._crash_tail
                if not tail and self._proc.stderr is not None:
                    try:
                        tail = self._proc.stderr.read().decode(errors="replace")
                    except OSError:
                        tail = ""
                raise LlamaServerError(f"llama-server exited early: {tail}")
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    if resp.status == 200:
                        logger.info(
                            "model %s ready (gguf=%s, url=%s)",
                            self.model.name,
                            str(self.model.resolve()),
                            self.base_url,
                        )
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        raise LlamaServerError(f"llama-server not healthy at {url} within {timeout_s}s")

    def stop(self) -> None:
        if self._proc is None:
            return
        self._watchdog_stop.set()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)
        self._proc = None
        self._watchdog_thread = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def completion(
        self,
        prompt: str,
        n_predict: int | None = None,
        id_slot: int = -1,
        *,
        kv_token_budget: int | None = None,
        kv_bind_req: Any | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompt: bool | None = None,
        current_pos: int | None = None,
        prefill_cancel: Any | None = None,
    ) -> dict[str, Any]:
        del kv_token_budget, kv_bind_req, kv_block_size, current_pos
        if self._proc is None or self._proc.poll() is not None:
            raise LlamaServerError("llama-server is not running")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "stream": False,
        }
        if n_predict is not None and n_predict > 0:
            payload["n_predict"] = n_predict
        if id_slot >= 0:
            payload["id_slot"] = id_slot
        # L3: persist prefix KV into pinned slot (pairs with cache_bridge derive_slot_id).
        if cache_prompt is not None:
            payload["cache_prompt"] = cache_prompt
        apply_sampler_to_completion_payload(payload, sampler)
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        try:
            conn, resp = cancellable_http_post(
                self.base_url,
                "/completion",
                body,
                headers,
                prefill_cancel=prefill_cancel,
            )
            if resp.status >= 400:
                raw = read_response_body(conn, resp, prefill_cancel=prefill_cancel)
                raise LlamaServerError(
                    f"llama-server /completion HTTP {resp.status}: "
                    f"{raw.decode(errors='replace')[:500]}"
                )
            raw = read_response_body(conn, resp, prefill_cancel=prefill_cancel)
        except LlamaServerError:
            raise
        except Exception as e:
            from runtime.kv.native_decode_loop import PrefillAbortedError

            if isinstance(e, PrefillAbortedError):
                raise
            raise LlamaServerError(f"llama-server /completion: {e}") from e
        try:
            data: dict[str, Any] = json.loads(raw.decode(errors="replace"))
        except json.JSONDecodeError as e:
            raise LlamaServerError(
                f"llama-server /completion returned non-JSON: {raw[:500]!r}"
            ) from e
        if "content" not in data and "response" in data:
            data["content"] = data["response"]
        return data

    def completion_stream(
        self,
        prompt: str,
        n_predict: int | None = None,
        id_slot: int = -1,
        *,
        kv_token_budget: int | None = None,
        kv_bind_req: Any | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompt: bool | None = None,
        current_pos: int | None = None,
        prefill_cancel: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        del kv_token_budget, kv_bind_req, kv_block_size, current_pos
        if self._proc is None or self._proc.poll() is not None:
            raise LlamaServerError("llama-server is not running")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "stream": True,
        }
        if n_predict is not None and n_predict > 0:
            payload["n_predict"] = n_predict
        if id_slot >= 0:
            payload["id_slot"] = id_slot
        # L3: persist prefix KV into pinned slot (pairs with cache_bridge derive_slot_id).
        if cache_prompt is not None:
            payload["cache_prompt"] = cache_prompt
        apply_sampler_to_completion_payload(payload, sampler)
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        conn, resp = cancellable_http_post(
            self.base_url,
            "/completion",
            body,
            headers,
            prefill_cancel=prefill_cancel,
        )
        if resp.status >= 400:
            raw = read_response_body(conn, resp, prefill_cancel=prefill_cancel)
            raise LlamaServerError(
                f"llama-server /completion HTTP {resp.status}: "
                f"{raw.decode(errors='replace')[:500]}"
            )
        try:
            yield from iter_llama_sse_lines(
                iter_response_chunks(conn, resp, prefill_cancel=prefill_cancel)
            )
        except Exception as e:
            from runtime.kv.native_decode_loop import PrefillAbortedError

            if isinstance(e, PrefillAbortedError):
                raise
            raise LlamaServerError(f"llama-server /completion stream: {e}") from e

    def completions_parallel(
        self,
        prompts: list[str],
        n_predict: int | None = None,
        *,
        id_slots: list[int] | None = None,
        kv_token_budgets: list[int] | None = None,
        kv_bind_reqs: list[Any] | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompts: list[bool] | None = None,
        current_positions: list[int | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run completions on distinct llama-server slots in parallel."""
        del kv_token_budgets, kv_bind_reqs, kv_block_size, current_positions
        if not prompts:
            return []
        slots = id_slots if id_slots is not None else list(range(len(prompts)))

        def _slot(idx: int) -> int:
            if idx < len(slots):
                return slots[idx]
            return idx

        def _cache_prompt(idx: int) -> bool | None:
            if cache_prompts is None or idx >= len(cache_prompts):
                return None
            return cache_prompts[idx]

        if len(prompts) == 1:
            return [
                self.completion(
                    prompts[0],
                    n_predict=n_predict,
                    id_slot=_slot(0),
                    sampler=sampler,
                    cache_prompt=_cache_prompt(0),
                )
            ]

        results: list[dict[str, Any] | None] = [None] * len(prompts)

        def _one(idx: int, prompt: str) -> tuple[int, dict[str, Any]]:
            return idx, self.completion(
                prompt,
                n_predict=n_predict,
                id_slot=_slot(idx),
                sampler=sampler,
                cache_prompt=_cache_prompt(idx),
            )

        workers = min(len(prompts), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, i, p) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                idx, data = fut.result()
                results[idx] = data
        return [r if r is not None else {} for r in results]

    def completions_parallel_stream(
        self,
        prompts: list[str],
        n_predict: int | None = None,
        *,
        id_slots: list[int] | None = None,
        kv_token_budgets: list[int] | None = None,
        kv_bind_reqs: list[Any] | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompts: list[bool] | None = None,
        current_positions: list[int | None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream completions on distinct llama-server slots (sequential fallback)."""
        del kv_token_budgets, kv_bind_reqs, kv_block_size, current_positions
        if not prompts:
            return iter(())
        slots = id_slots if id_slots is not None else list(range(len(prompts)))

        def _slot(idx: int) -> int:
            if idx < len(slots):
                return slots[idx]
            return idx

        def _cache_prompt(idx: int) -> bool | None:
            if cache_prompts is None or idx >= len(cache_prompts):
                return None
            return cache_prompts[idx]

        def _sequential() -> Iterator[dict[str, Any]]:
            for idx, prompt in enumerate(prompts):
                for chunk in self.completion_stream(
                    prompt,
                    n_predict=n_predict,
                    id_slot=_slot(idx),
                    sampler=sampler,
                    cache_prompt=_cache_prompt(idx),
                ):
                    out = dict(chunk)
                    out.setdefault("seq_idx", idx)
                    out.setdefault("seq_id", _slot(idx))
                    yield out

        return _sequential()
