#!/usr/bin/env bash
# Download MoringLabs GPT-OSS-120B MLX 3.6bit and import into zerollama (Option B).
set -euo pipefail

REPO="MoringLabs/GPT-OSS-120B-MLX-3.6bit"
MODEL_NAME="${1:-gpt-oss-120b-mlx-3.6bit}"
IMPORT_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}/.import/gpt-oss-120b-mlx-3.6bit"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZERO="${ZERO:-$ROOT/zerollama}"

log() { printf '[import-gpt-oss-mlx] %s\n' "$*"; }

need_gb() {
  python3 - <<'PY'
import shutil, os
path = os.path.expanduser(os.environ.get("OLLAMA_MODELS", "~/.ollama/models"))
usage = shutil.disk_usage(path)
print(f"avail_gb={usage.free / (1024**3):.1f}")
PY
}

log "Disk before download: $(need_gb)"
log "Downloading $REPO → $IMPORT_DIR (~48 GiB)"

export IMPORT_REPO="$REPO" IMPORT_DIR="$IMPORT_DIR"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
python3 -u - <<'PY'
import os
import time
# Must set before huggingface_hub import (xet stalls on large shards for some networks).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

from huggingface_hub import hf_hub_download, list_repo_files

repo = os.environ["IMPORT_REPO"]
dest = os.path.expanduser(os.environ["IMPORT_DIR"])
max_retries = int(os.environ.get("HF_DOWNLOAD_RETRIES", "8"))
retry_sleep = int(os.environ.get("HF_DOWNLOAD_RETRY_SLEEP", "30"))
os.makedirs(dest, exist_ok=True)

def log(msg: str) -> None:
    print(f"[import-gpt-oss-mlx] {msg}", flush=True)

def dir_size(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024**3)

files = list_repo_files(repo)
log(f"{len(files)} repo files; downloading sequentially (resumes partial shards)")
for i, name in enumerate(files, 1):
    out = os.path.join(dest, name)
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        log(f"[{i}/{len(files)}] skip existing {name} ({os.path.getsize(out) / (1024**3):.2f} GiB)")
        continue
    log(f"[{i}/{len(files)}] fetching {name}")
    for attempt in range(1, max_retries + 1):
        try:
            path = hf_hub_download(repo, name, local_dir=dest, local_dir_use_symlinks=False)
            break
        except Exception as err:
            if attempt >= max_retries:
                raise
            log(f"[{i}/{len(files)}] retry {attempt}/{max_retries} after error: {err}")
            time.sleep(retry_sleep * attempt)
    log(f"[{i}/{len(files)}] done {name} ({os.path.getsize(path) / (1024**3):.2f} GiB); total={dir_size(dest):.2f} GiB")

log(f"download complete: {dest} ({dir_size(dest):.2f} GiB)")
PY

log "Download complete. Disk: $(need_gb)"

avail=$(python3 - <<'PY'
import shutil, os
path = os.path.expanduser(os.environ.get("OLLAMA_MODELS", "~/.ollama/models"))
print(int(shutil.disk_usage(path).free / (1024**3)))
PY
)
if [ "$avail" -lt 95 ]; then
  log "WARNING: ${avail} GiB free — zerollama create needs ~48 GiB blobs on top of the ${IMPORT_DIR} tree (~48 GiB)."
  log "Free space first (e.g. zerollama rm gpt-oss:120b / old GGUF) or delete $IMPORT_DIR after a failed create."
fi
log "Importing into zerollama as $MODEL_NAME"

MODEFILE="$(mktemp)"
trap 'rm -f "$MODEFILE"' EXIT
cat >"$MODEFILE" <<EOF
FROM $IMPORT_DIR
EOF

"$ZERO" create "$MODEL_NAME" --experimental -f "$MODEFILE"

log "Import complete: $MODEL_NAME"
log "Verify: $ZERO list | grep -F '$MODEL_NAME' || true"
log "Note: inference needs GptOssForCausalLM in mlxrunner (not yet registered)."
log "Optional: rm -rf '$IMPORT_DIR' after import to reclaim ~48 GiB source tree."
