"""Subprocess wrapper for llama-server (GGUF inference)."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from runtime.logutil import get_logger
from runtime.worker.sampler_options import (
    SamplerOptions,
    apply_sampler_to_completion_payload,
)
from runtime.worker.llama_stream import iter_llama_sse_lines

logger = get_logger("llama_server")


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

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

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

        env = os.environ.copy()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._wait_healthy(timeout_s=120.0)

    def _wait_healthy(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                err = self._proc.stderr.read().decode() if self._proc.stderr else ""
                raise LlamaServerError(f"llama-server exited early: {err}")
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
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None

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
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600.0) as resp:
                raw = resp.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace") if e.fp else ""
            raise LlamaServerError(
                f"llama-server /completion HTTP {e.code}: {err_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise LlamaServerError(f"llama-server /completion: {e}") from e
        try:
            data: dict[str, Any] = json.loads(raw)
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
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=600.0)
        try:
            yield from iter_llama_sse_lines(resp)
        finally:
            resp.close()

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
