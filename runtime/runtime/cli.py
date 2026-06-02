"""CLI entry: zerollama-runtime serve."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="zerollama-runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="run HTTP sidecar")
    serve_p.add_argument("--host", default=None)
    serve_p.add_argument("--port", type=int, default=None)
    serve_p.add_argument(
        "--config",
        default=None,
        help="YAML topology (default: ZEROLLAMA_RUNTIME_CONFIG or configs/dual_4090.yaml)",
    )

    up_p = sub.add_parser(
        "up",
        help="start runtime sidecar then zerollama serve (sets ZEROLLAMA_RUNTIME_URL)",
    )
    up_p.add_argument("--config", default=None)
    up_p.add_argument("--runtime-host", default="127.0.0.1")
    up_p.add_argument("--runtime-port", type=int, default=8081)
    up_p.add_argument(
        "--zerollama",
        default=None,
        help="path to zerollama binary (default: PATH)",
    )

    args = parser.parse_args()
    if args.command == "serve":
        _serve(args)
    elif args.command == "up":
        _up(args)


def _serve(args: argparse.Namespace) -> None:
    from runtime.config import RuntimeConfig
    from runtime.logutil import setup_logging
    from runtime.server.app import create_app

    import os
    from pathlib import Path

    setup_logging()

    from runtime.vram_env_apply import apply_exported_vram_env
    from runtime.vram_yaml_defaults import apply_vram_defaults_from_config

    if args.host:
        os.environ["ZEROLLAMA_RUNTIME_HOST"] = args.host
    if args.port:
        os.environ["ZEROLLAMA_RUNTIME_PORT"] = str(args.port)
    if args.config:
        os.environ["ZEROLLAMA_RUNTIME_CONFIG"] = args.config
    elif not os.environ.get("ZEROLLAMA_RUNTIME_CONFIG"):
        from runtime.autoconfig import resolve_default_config_path

        os.environ["ZEROLLAMA_RUNTIME_CONFIG"] = str(resolve_default_config_path())
    # WHY order: single_gpu.yaml vram: defaults, then optional exported factor file.
    apply_vram_defaults_from_config()
    apply_exported_vram_env()
    cfg = RuntimeConfig.from_env()

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("pip install -e '.[serve]'") from e

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


def _up(args: argparse.Namespace) -> None:
    from pathlib import Path

    from runtime.supervisor import SupervisorError, run_stack

    cfg = Path(args.config) if args.config else None
    try:
        code = run_stack(
            zerollama_bin=args.zerollama,
            config=cfg,
            runtime_host=args.runtime_host,
            runtime_port=args.runtime_port,
        )
    except SupervisorError as e:
        raise SystemExit(str(e)) from e
    raise SystemExit(code)


if __name__ == "__main__":
    main()
