"""Start the FastAPI runtime on a background thread (shared CPython with training)."""

from __future__ import annotations

import threading
from typing import Optional

_thread: Optional[threading.Thread] = None
_port: int = 0


def start_embedded_server(port: int = 8081) -> None:
    global _thread, _port
    if _thread is not None and _thread.is_alive():
        return
    _port = port

    def _run() -> None:
        from runtime.config import RuntimeConfig
        from runtime.server.app import create_app

        try:
            import uvicorn
        except ImportError as e:
            raise RuntimeError("pip install -e 'runtime/.[serve]'") from e

        from runtime.logutil import get_logger

        cfg = RuntimeConfig.from_env()
        log = get_logger("embed")
        if cfg.llama_model and cfg.llama_model.is_file():
            log.info(
                "embedded runtime starting; default gguf model %s (%s)",
                cfg.llama_model.name,
                cfg.llama_model.resolve(),
            )
        else:
            log.warning(
                "embedded runtime starting; no LLAMA_MODEL set (configure before inference)"
            )
        app = create_app(cfg)
        uvicorn.run(
            app,
            host=cfg.host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    _thread = threading.Thread(
        target=_run,
        name="zerollama-runtime-embed",
        daemon=True,
    )
    _thread.start()


def embedded_port() -> int:
    return _port
