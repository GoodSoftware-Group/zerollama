#!/usr/bin/env python3
"""Lab-resident TRELLIS.2 process: load once, bake many stills.

Not a zerollama sidecar. Never binds 11434 or 8081.
Default lab port: 127.0.0.1:18091 (JSONL request / response).

ATTN_BACKEND stays sdpa — microbench showed fused sparse_attention_fwd
slower than native MPS SDPA on TRELLIS.2 shapes (see lab/out/attn_microbench.json).

Jobs are serialized: one pipeline.run / bake at a time (MPS is not multi-caller).

Usage:
  source ~/.zerollama/third_party/trellis-mac/env-mac.sh
  python scripts/mesh/trellis_mesh_lab_mac.py --host 127.0.0.1 --port 18091

  # one job (after ready):
  printf '%s\\n' '{"image":".../still.png","out":".../a.glb","seed":1}' \\
    | nc 127.0.0.1 18091

Or JSONL over stdin with --stdio.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

# Import generate_mac after MPS env defaults (module sets ATTN_BACKEND=sdpa).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trellis_mesh_generate import (  # noqa: E402
    _install_dinov3_mirror,
    _install_rembg_stub_or_mirror,
)
from trellis_mesh_generate_mac import (  # noqa: E402
    _TRELLIS_ROOT,
    _progress,
    _require_metal_baker,
    generate_one,
)

_BLOCKED_PORTS = {11434, 8081}
_JOB_LOCK = threading.Lock()


def _load_pipeline(model: str):
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available")
    baker = _require_metal_baker()
    _progress(f"baker={baker}")
    # Same gated-repo remaps as the one-shot wrapper (before from_pretrained).
    dinov3 = _install_dinov3_mirror()
    rembg = _install_rembg_stub_or_mirror(stub=True)
    _progress(f"dinov3_model={dinov3} rembg={rembg}")
    _progress(f"load_model={model} (resident; once)")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model)
    pipeline.to(torch.device("mps"))
    _progress(f"load_sec={time.time() - t0:.1f}")
    return pipeline


def _handle_job(pipeline, req: dict, *, loaded_model: str, default_out_dir: Path) -> dict:
    image = req.get("image")
    if not image:
        return {"ok": False, "exit_code": 2, "error": "missing image"}
    req_model = req.get("model")
    if req_model and str(req_model) != loaded_model:
        return {
            "ok": False,
            "exit_code": 2,
            "error": f"model mismatch: resident loaded {loaded_model!r}, request has {req_model!r}",
            "loaded_model": loaded_model,
        }
    out = req.get("out")
    if not out:
        stem = Path(image).stem
        out = str(default_out_dir / f"{stem}-mps.glb")

    # Serialize: MPS + this pipeline are one-caller-at-a-time.
    with _JOB_LOCK:
        try:
            code, note = generate_one(
                image_path=Path(image),
                out_path=Path(out),
                pipeline=pipeline,
                keep_pipeline=True,
                resolution=int(req.get("resolution", 512)),
                seed=int(req.get("seed", 1)),
                decimation=int(req.get("decimation", 100_000)),
                texture_size=int(req.get("texture_size", 1024)),
                model=loaded_model,
                preprocess_image=bool(req.get("preprocess_image", False)),
                note_out=str(req.get("note_out", "")),
                skip_bake=bool(req.get("skip_bake", False)),
                allow_fragmented=bool(req.get("allow_fragmented", False)),
                prune_min_faces=int(req.get("prune_min_faces", 100)),
                keep_largest=int(req.get("keep_largest", 8)),
                resident=True,
            )
        except Exception as e:
            return {"ok": False, "exit_code": 2, "error": str(e), "loaded_model": loaded_model}

    return {
        "ok": code == 0,
        "exit_code": code,
        "out": note.get("out"),
        "elapsed_sec": note.get("elapsed_sec"),
        "load_sec": note.get("load_sec"),
        "generate_sec": note.get("generate_sec"),
        "bake_sec": note.get("bake_sec"),
        "alphaMode": note.get("alphaMode"),
        "topology": note.get("topology"),
        "errors": note.get("errors"),
        "error": note.get("error"),
        "attn_backend": note.get("attn_backend"),
        "resident": True,
        "loaded_model": loaded_model,
        "reused_pipeline": True,
    }


def _is_quit(line: str, req: dict | None) -> bool:
    if line in ("quit", "exit"):
        return True
    if req and req.get("cmd") in ("quit", "exit"):
        return True
    return False


def _serve_stdio(pipeline, *, loaded_model: str, default_out_dir: Path) -> int:
    _progress("ready stdio jsonl")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line) if line.startswith("{") else None
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "exit_code": 2, "error": str(e)}), flush=True)
            continue
        if _is_quit(line, req):
            print(json.dumps({"ok": True, "bye": True}), flush=True)
            return 0
        if req is None:
            print(
                json.dumps({"ok": False, "exit_code": 2, "error": "expected JSON object"}),
                flush=True,
            )
            continue
        if req.get("cmd") in ("ping", "health"):
            print(
                json.dumps({"ok": True, "ready": True, "loaded_model": loaded_model}),
                flush=True,
            )
            continue
        if req.get("cmd") == "shutdown":
            print(json.dumps({"ok": True, "bye": True}), flush=True)
            return 0
        resp = _handle_job(
            pipeline, req, loaded_model=loaded_model, default_out_dir=default_out_dir
        )
        print(json.dumps(resp), flush=True)
    return 0


def _serve_tcp(
    pipeline, *, host: str, port: int, loaded_model: str, default_out_dir: Path
) -> int:
    if port in _BLOCKED_PORTS:
        print(
            f"error: port {port} is reserved for zerollama production; use a lab port (default 18091)",
            file=sys.stderr,
        )
        return 2
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    # backlog small; jobs are serialized by _JOB_LOCK anyway
    sock.listen(2)
    _progress(f"ready tcp://{host}:{port} jsonl (lab only; not zerollama)")
    shutdown_event = threading.Event()

    def _client(conn: socket.socket, addr) -> None:
        try:
            with conn:
                buf = b""
                while not shutdown_event.is_set():
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            req = json.loads(line) if line.startswith("{") else None
                        except json.JSONDecodeError as e:
                            conn.sendall(
                                (json.dumps({"ok": False, "exit_code": 2, "error": str(e)}) + "\n").encode()
                            )
                            continue
                        if _is_quit(line, req):
                            # Drop this client only; process stays up for more stills.
                            conn.sendall(b'{"ok":true,"bye":true,"scope":"connection"}\n')
                            return
                        if req is None:
                            conn.sendall(
                                b'{"ok":false,"exit_code":2,"error":"expected JSON object"}\n'
                            )
                            continue
                        if req.get("cmd") in ("ping", "health"):
                            conn.sendall(
                                (
                                    json.dumps(
                                        {
                                            "ok": True,
                                            "ready": True,
                                            "loaded_model": loaded_model,
                                        }
                                    )
                                    + "\n"
                                ).encode()
                            )
                            continue
                        if req.get("cmd") == "shutdown":
                            conn.sendall(b'{"ok":true,"bye":true,"scope":"process"}\n')
                            shutdown_event.set()
                            return
                        resp = _handle_job(
                            pipeline,
                            req,
                            loaded_model=loaded_model,
                            default_out_dir=default_out_dir,
                        )
                        conn.sendall((json.dumps(resp) + "\n").encode())
        except Exception as e:
            _progress(f"client {addr} error: {e}")

    try:
        sock.settimeout(1.0)
        while not shutdown_event.is_set():
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            threading.Thread(target=_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        _progress("shutdown")
        return 0
    finally:
        sock.close()
    _progress("shutdown")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("TRELLIS_LAB_PORT", "18091")),
        help="Lab TCP port (default 18091). Never 11434 or 8081.",
    )
    ap.add_argument(
        "--stdio",
        action="store_true",
        help="JSONL over stdin/stdout instead of TCP",
    )
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Default output dir when request omits out",
    )
    args = ap.parse_args()

    if args.port in _BLOCKED_PORTS:
        print(f"error: refused production port {args.port}", file=sys.stderr)
        return 2

    root = Path(_TRELLIS_ROOT)
    default_out = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else root / "lab" / "out"
    )
    default_out.mkdir(parents=True, exist_ok=True)
    _progress(f"trellis_root={root} out_dir={default_out}")
    _progress(f"attn_backend={os.environ.get('ATTN_BACKEND', 'sdpa')}")

    try:
        pipeline = _load_pipeline(args.model)
    except Exception as e:
        print(f"error: failed to load pipeline: {e}", file=sys.stderr)
        return 2

    if args.stdio:
        return _serve_stdio(pipeline, loaded_model=args.model, default_out_dir=default_out)
    return _serve_tcp(
        pipeline,
        host=args.host,
        port=args.port,
        loaded_model=args.model,
        default_out_dir=default_out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
