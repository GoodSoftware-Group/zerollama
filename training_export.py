"""Post-train export: Modelfile+ADAPTER register and optional GGUF convert (T7).

Why this exists: operators previously had to hand-write a Modelfile after
``lora_adapter/`` landed. Unsloth's train→GGUF loop is the product expectation;
zerollama's native path is FROM+ADAPTER (same weights, no merge) with optional
merge→convert_hf_to_gguf→llama-quantize for a standalone GGUF tag.

Register prefers ``zerollama create`` CLI; if the binary is missing (or
``register_via=http``), falls back to blob upload + ``POST /api/create``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


ProgressFn = Callable[[float, str], None]
UnloadFn = Callable[[], None]


def _progress(cb: Optional[ProgressFn], pct: float, msg: str) -> None:
    if cb:
        cb(pct, msg)


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def find_convert_hf_to_gguf() -> Optional[Path]:
    env = os.environ.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP_ROOT")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env).expanduser() / "convert_hf_to_gguf.py")
    root = repo_root()
    candidates.extend(
        [
            root / "llama" / "llama.cpp" / "convert_hf_to_gguf.py",
            root.parent / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def find_llama_quantize() -> Optional[Path]:
    which = shutil.which("llama-quantize")
    if which:
        return Path(which)
    env = os.environ.get("LLAMA_CPP_DIR") or os.environ.get("LLAMA_CPP_ROOT")
    candidates: List[Path] = []
    if env:
        base = Path(env).expanduser()
        candidates.extend(
            [
                base / "build" / "bin" / "llama-quantize",
                base / "build" / "bin" / "Release" / "llama-quantize",
            ]
        )
    root = repo_root()
    candidates.extend(
        [
            root / "llama" / "llama.cpp" / "build" / "bin" / "llama-quantize",
            root.parent / "llama.cpp" / "build" / "bin" / "llama-quantize",
        ]
    )
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def find_zerollama_bin() -> Optional[Path]:
    env = os.environ.get("ZEROLLAMA_BIN")
    if env and Path(env).is_file():
        return Path(env)
    # Prefer repo-local build over a stale PATH install.
    root = repo_root()
    for name in ("zerollama", "ollama"):
        p = root / name
        if p.is_file() and os.access(p, os.X_OK):
            return p
    which = shutil.which("zerollama") or shutil.which("ollama")
    if which:
        return Path(which)
    return None


def normalize_quant(q: str) -> str:
    q = (q or "q4_k_m").strip().lower().replace("-", "_")
    aliases = {
        "f16": "f16",
        "fp16": "f16",
        "q8_0": "q8_0",
        "q8": "q8_0",
        "q4_k_m": "q4_k_m",
        "q4km": "q4_k_m",
        "q4_0": "q4_0",
        "q5_k_m": "q5_k_m",
    }
    return aliases.get(q, q)


def quantize_type_arg(q: str) -> Optional[str]:
    """Return llama-quantize TYPE or None if convert --outtype is enough (f16)."""
    q = normalize_quant(q)
    if q == "f16":
        return None
    return q.upper()  # Q4_K_M, Q8_0, …


def write_adapter_modelfile(
    *,
    from_model: str,
    adapter_path: str | Path,
    dest: str | Path,
    system: str = "",
) -> Path:
    """Write a Modelfile that serves the PEFT adapter against a base FROM."""
    adapter_path = Path(adapter_path).resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"FROM {from_model}", f"ADAPTER {adapter_path}"]
    if system:
        lines.append(f'SYSTEM """{system}"""')
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def write_gguf_modelfile(
    *,
    gguf_path: str | Path,
    dest: str | Path,
    system: str = "",
) -> Path:
    gguf_path = Path(gguf_path).resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"FROM {gguf_path}"]
    if system:
        lines.append(f'SYSTEM """{system}"""')
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def parse_modelfile_simple(text: str, *, relative_dir: str | Path) -> Dict[str, Any]:
    """Parse FROM / ADAPTER / SYSTEM from a Modelfile (T7 HTTP create)."""
    relative_dir = Path(relative_dir)
    from_arg = ""
    adapter_arg = ""
    system = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("FROM "):
            from_arg = line[5:].strip().strip('"').strip("'")
        elif upper.startswith("ADAPTER "):
            adapter_arg = line[8:].strip().strip('"').strip("'")
        elif upper.startswith("SYSTEM "):
            rest = line[7:].strip()
            if rest.startswith('"""') and rest.endswith('"""') and len(rest) >= 6:
                system = rest[3:-3]
            else:
                system = rest.strip('"').strip("'")
    out: Dict[str, Any] = {"from": from_arg, "system": system}
    if adapter_arg:
        p = Path(adapter_arg)
        if not p.is_absolute():
            p = (relative_dir / p).resolve()
        out["adapter"] = str(p)
    if from_arg:
        fp = Path(from_arg)
        if not fp.is_absolute():
            cand = (relative_dir / fp).resolve()
            if cand.is_file():
                out["from_file"] = str(cand)
        elif fp.is_file():
            out["from_file"] = str(fp)
    return out


def file_sha256_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def collect_path_digests(path: Path) -> Dict[str, str]:
    """Map absolute file paths → sha256 digests (files under a dir, or one file)."""
    path = Path(path)
    out: Dict[str, str] = {}
    if path.is_file():
        out[str(path.resolve())] = file_sha256_digest(path)
        return out
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            out[str(fp.resolve())] = file_sha256_digest(fp)
    return out


def ollama_host() -> str:
    host = (
        os.environ.get("OLLAMA_HOST")
        or os.environ.get("ZEROLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).strip()
    if host.startswith(":"):
        host = f"http://127.0.0.1{host}"
    elif "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def _http_request(
    method: str,
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 600.0,
) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        return int(e.code), body


def upload_blob(host: str, digest: str, path: Path, *, timeout: float = 600.0) -> None:
    # HEAD first — skip upload when blob already present.
    code, _ = _http_request("HEAD", f"{host}/api/blobs/{digest}", timeout=30.0)
    if code == 200:
        return
    with open(path, "rb") as f:
        payload = f.read()
    code, body = _http_request(
        "POST",
        f"{host}/api/blobs/{digest}",
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        timeout=timeout,
    )
    if code not in (200, 201):
        raise RuntimeError(
            f"blob upload {digest} failed HTTP {code}: {body.decode('utf-8', errors='replace')[:500]}"
        )


def register_model_http(
    model_name: str,
    modelfile_path: str | Path,
    *,
    host: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Upload blobs + ``POST /api/create`` (no CLI required)."""
    modelfile_path = Path(modelfile_path)
    host = (host or ollama_host()).rstrip("/")
    parsed = parse_modelfile_simple(
        modelfile_path.read_text(encoding="utf-8"),
        relative_dir=modelfile_path.parent,
    )
    req: Dict[str, Any] = {"model": model_name, "stream": False}
    if parsed.get("system"):
        req["system"] = parsed["system"]

    files_map: Dict[str, str] = {}
    adapters_map: Dict[str, str] = {}

    if parsed.get("adapter"):
        digests = collect_path_digests(Path(parsed["adapter"]))
        _progress(progress, 96.0, f"Uploading {len(digests)} adapter blob(s) to {host}")
        for fpath, digest in digests.items():
            upload_blob(host, digest, Path(fpath))
            adapters_map[Path(fpath).name] = digest
        req["adapters"] = adapters_map
        req["from"] = parsed.get("from") or ""
        if not req["from"]:
            return {"status": "error", "error": "ADAPTER Modelfile missing FROM"}
    elif parsed.get("from_file"):
        digests = collect_path_digests(Path(parsed["from_file"]))
        _progress(progress, 96.0, f"Uploading {len(digests)} GGUF blob(s) to {host}")
        for fpath, digest in digests.items():
            upload_blob(host, digest, Path(fpath))
            files_map[Path(fpath).name] = digest
        req["files"] = files_map
    else:
        # FROM is a registry/model tag — no local files.
        req["from"] = parsed.get("from") or ""
        if not req["from"]:
            return {"status": "error", "error": "Modelfile missing FROM"}

    _progress(progress, 97.0, f"POST {host}/api/create model={model_name}")
    body = json.dumps(req).encode("utf-8")
    code, resp = _http_request(
        "POST",
        f"{host}/api/create",
        data=body,
        headers={"Content-Type": "application/json"},
        timeout=float(os.environ.get("ZEROLLAMA_TRAIN_CREATE_TIMEOUT", "600")),
    )
    text = resp.decode("utf-8", errors="replace")
    if code not in (200, 201):
        return {
            "status": "error",
            "error": f"HTTP {code}: {text[:800]}",
            "host": host,
            "request_keys": sorted(req.keys()),
        }
    return {
        "status": "ok",
        "model": model_name,
        "modelfile": str(modelfile_path),
        "via": "http",
        "host": host,
    }


def register_model_cli(
    model_name: str,
    modelfile_path: str | Path,
    *,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Run ``zerollama create NAME -f Modelfile`` (uploads ADAPTER blobs)."""
    bin_path = find_zerollama_bin()
    if not bin_path:
        return {
            "status": "skipped",
            "error": "zerollama/ollama binary not found",
            "modelfile": str(modelfile_path),
        }
    _progress(progress, 96.0, f"Registering {model_name} via {bin_path.name} create")
    cmd = [str(bin_path), "create", model_name, "-f", str(modelfile_path)]
    # Point CLI at same host as HTTP fallback would use.
    env = os.environ.copy()
    if "OLLAMA_HOST" not in env and "ZEROLLAMA_HOST" in env:
        env["OLLAMA_HOST"] = env["ZEROLLAMA_HOST"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("ZEROLLAMA_TRAIN_CREATE_TIMEOUT", "600")),
            check=False,
            env=env,
        )
    except Exception as e:
        return {"status": "error", "error": str(e), "cmd": cmd}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        return {"status": "error", "error": err, "cmd": cmd, "via": "cli"}
    return {
        "status": "ok",
        "model": model_name,
        "modelfile": str(modelfile_path),
        "cmd": cmd,
        "via": "cli",
    }


def register_model(
    model_name: str,
    modelfile_path: str | Path,
    *,
    progress: Optional[ProgressFn] = None,
    via: str = "auto",
) -> Dict[str, Any]:
    """Register via CLI and/or HTTP.

    ``via``: ``auto`` (CLI then HTTP), ``cli``, ``http``.
    """
    via = (via or "auto").strip().lower()
    if via == "http":
        return register_model_http(model_name, modelfile_path, progress=progress)
    if via == "cli":
        return register_model_cli(model_name, modelfile_path, progress=progress)

    cli = register_model_cli(model_name, modelfile_path, progress=progress)
    if cli.get("status") == "ok":
        return cli
    # Binary missing or create failed → try HTTP (serve may still be up).
    http = register_model_http(model_name, modelfile_path, progress=progress)
    if http.get("status") == "ok":
        http["cli_fallback"] = cli
        return http
    return {
        "status": "error",
        "error": f"cli: {cli.get('error')}; http: {http.get('error')}",
        "cli": cli,
        "http": http,
        "modelfile": str(modelfile_path),
    }


def release_training_memory(
    *,
    unload_fn: Optional[UnloadFn] = None,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Free GPU/CPU training weights before merge/convert (T7 memory-cap)."""
    info: Dict[str, Any] = {"unloaded": False, "cache_cleared": False}
    _progress(progress, 90.5, "Releasing training VRAM before GGUF convert")
    if unload_fn is not None:
        try:
            unload_fn()
            info["unloaded"] = True
        except Exception as e:
            info["unload_error"] = str(e)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            info["cache_cleared"] = True
            info["cuda_mem_allocated"] = int(torch.cuda.memory_allocated())
    except Exception as e:
        info["cache_error"] = str(e)
    return info


def merge_and_save_hf(
    model: Any,
    tokenizer: Any,
    merged_dir: str | Path,
    *,
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Merge PEFT adapters into base weights and save an HF directory."""
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress, 91.0, "Merging LoRA into base weights")
    to_save = model
    merge = getattr(model, "merge_and_unload", None)
    if callable(merge):
        to_save = merge()
    _progress(progress, 92.0, f"Saving merged HF → {merged_dir}")
    to_save.save_pretrained(str(merged_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(merged_dir))
    return merged_dir


def convert_hf_to_gguf(
    hf_dir: str | Path,
    outfile: str | Path,
    *,
    outtype: str = "f16",
    progress: Optional[ProgressFn] = None,
) -> Path:
    script = find_convert_hf_to_gguf()
    if not script:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found — set LLAMA_CPP_DIR or clone llama.cpp beside the repo"
        )
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    _progress(progress, 93.0, f"Converting HF → GGUF ({outtype})")
    cmd = [
        sys.executable,
        str(script),
        str(hf_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    env = os.environ.copy()
    # Soft memory hint for convert subprocess (operators can tighten).
    if "ZEROLLAMA_TRAIN_CONVERT_MAX_WORKERS" in os.environ:
        env["OMP_NUM_THREADS"] = os.environ["ZEROLLAMA_TRAIN_CONVERT_MAX_WORKERS"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"convert_hf_to_gguf failed: {err}")
    if not outfile.is_file():
        raise RuntimeError(f"convert_hf_to_gguf produced no file at {outfile}")
    return outfile


def quantize_gguf(
    src: str | Path,
    dest: str | Path,
    quant: str,
    *,
    progress: Optional[ProgressFn] = None,
) -> Path:
    qtype = quantize_type_arg(quant)
    if qtype is None:
        # f16 — already the convert output
        return Path(src)
    qbin = find_llama_quantize()
    if not qbin:
        raise FileNotFoundError(
            "llama-quantize not found — set LLAMA_CPP_DIR or build llama.cpp (build/bin/llama-quantize)"
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _progress(progress, 94.0, f"Quantizing GGUF → {qtype}")
    cmd = [str(qbin), str(src), str(dest), qtype]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"llama-quantize failed: {err}")
    if not dest.is_file():
        raise RuntimeError(f"llama-quantize produced no file at {dest}")
    return dest


def default_register_name(output_dir: str | Path, model_name: str) -> str:
    base = Path(output_dir).name.strip() or "finetune"
    # sanitize tag
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in base).strip("-")
    if not safe:
        safe = "finetune"
    return f"{safe}:latest"


def resolve_export_unload(request: Dict[str, Any]) -> bool:
    """Default: unload training weights before GGUF convert (memory-cap)."""
    if "export_unload" in request:
        return bool(request.get("export_unload"))
    # env override
    env = os.environ.get("ZEROLLAMA_TRAIN_EXPORT_UNLOAD", "").strip().lower()
    if env in ("0", "false", "off", "no"):
        return False
    if env in ("1", "true", "on", "yes"):
        return True
    # Only needed when merging/converting (holds a second full copy of weights).
    return bool(request.get("export_gguf", False))


def run_export(
    *,
    request: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    model_name: str,
    output_dir: str | Path,
    adapter_path: str | Path,
    progress: Optional[ProgressFn] = None,
    unload_fn: Optional[UnloadFn] = None,
) -> Dict[str, Any]:
    """Execute optional register + GGUF export after a successful train save.

    Request fields (all optional):
      register_model: bool | str — if true, auto-name; if str, model tag
      register_via: auto|cli|http
      export_from: str — FROM for ADAPTER Modelfile (default: model_name)
      export_gguf: bool — merge + convert (+ quantize)
      export_quant: f16|q8_0|q4_k_m (default q4_k_m)
      export_gguf_dir: path (default: output_dir/gguf)
      export_system: optional SYSTEM line
      export_unload: bool — free training VRAM before merge/convert (default on for export_gguf)
    """
    out: Dict[str, Any] = {}
    output_dir = Path(output_dir)
    adapter_path = Path(adapter_path)

    register = request.get("register_model", False)
    export_gguf = bool(request.get("export_gguf", False))
    if not register and not export_gguf:
        return out

    export_from = str(request.get("export_from") or model_name).strip()
    export_system = str(request.get("export_system") or "").strip()
    quant = normalize_quant(str(request.get("export_quant") or "q4_k_m"))
    register_via = str(request.get("register_via") or "auto").strip().lower()

    if isinstance(register, str) and register.strip():
        reg_name = register.strip()
        do_register = True
    elif register is True:
        reg_name = default_register_name(output_dir, model_name)
        do_register = True
    else:
        reg_name = ""
        do_register = False

    gguf_path: Optional[Path] = None
    if export_gguf:
        gguf_dir = Path(request.get("export_gguf_dir") or (output_dir / "gguf"))
        gguf_dir.mkdir(parents=True, exist_ok=True)
        merged_dir = gguf_dir / "merged_hf"
        try:
            # Merge while model still in memory; then release before convert subprocess.
            merge_and_save_hf(model, tokenizer, merged_dir, progress=progress)
            if resolve_export_unload(request):
                out["memory_cap"] = release_training_memory(
                    unload_fn=unload_fn, progress=progress
                )
            f16_path = gguf_dir / "model.f16.gguf"
            convert_hf_to_gguf(merged_dir, f16_path, outtype="f16", progress=progress)
            if normalize_quant(quant) == "f16":
                gguf_path = f16_path
            else:
                q_path = gguf_dir / f"model.{normalize_quant(quant)}.gguf"
                gguf_path = quantize_gguf(f16_path, q_path, quant, progress=progress)
            out["gguf_path"] = str(gguf_path)
            out["export_quant"] = quant
            mf = write_gguf_modelfile(
                gguf_path=gguf_path,
                dest=gguf_dir / "Modelfile",
                system=export_system,
            )
            out["gguf_modelfile"] = str(mf)
        except Exception as e:
            out["gguf_error"] = str(e)
            _progress(progress, 94.0, f"GGUF export failed: {e}")

    # ADAPTER Modelfile always when registering without successful GGUF, or as sidecar.
    adapter_mf = write_adapter_modelfile(
        from_model=export_from,
        adapter_path=adapter_path,
        dest=output_dir / "Modelfile",
        system=export_system,
    )
    out["adapter_modelfile"] = str(adapter_mf)

    if do_register:
        # Prefer GGUF Modelfile when export succeeded; else ADAPTER overlay.
        mf_for_create = out.get("gguf_modelfile") or str(adapter_mf)
        reg = register_model(
            reg_name, mf_for_create, progress=progress, via=register_via
        )
        out["register"] = reg
        if reg.get("status") == "ok":
            out["registered_model"] = reg_name

    return out
