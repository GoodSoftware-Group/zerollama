"""Entry: python -m trainingdaemon --uds-path /path/to/sock (run with PYTHONPATH=.../x/trainingdaemon)."""

from __future__ import annotations

import argparse
import os

from .server import serve_uds


def main() -> None:
    p = argparse.ArgumentParser(description="Ollama internal training daemon (gRPC over UDS)")
    p.add_argument("--uds-path", required=True, help="Unix domain socket path for gRPC (no tcp://)")
    args = p.parse_args()
    path = os.path.abspath(args.uds_path)
    if os.path.exists(path):
        os.unlink(path)
    serve_uds(path)


if __name__ == "__main__":
    main()
