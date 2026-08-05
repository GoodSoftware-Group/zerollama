# Wan Pure-C client (`x/wan-c`)

Local text-to-video via a **strict C11** `wan-cli` that:

1. Reads **GGUF** weights (converted offline from Wan safetensors/`.pth`)
2. Uploads tensors into named **UmaBuffers** (`BUF_ALLOC` / `BUF_PUT`) — client-side only
3. Dispatches compute as **GRAPH** recipes to the machine **`uma_daemon`** compute broker
4. Encodes frames with system **`ffmpeg`**

## Prerequisites

- **`uma_daemon` running** (one per Mac). Do not start a second broker.
  - `cd …/uma_toolkit && make uma-daemon` or open `UMAStatus.app`
- Wan 2.1 T2V 1.3B checkpoint under `~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B`
- `ffmpeg` on `PATH`

## Install

```bash
./scripts/video/install_wan_c.sh
source ~/.zerollama/third_party/wan-c/env.sh
```

Install fails closed if `uma_daemon` is down (no silent local fallback for production path).

Lab-only without broker:

```bash
UMA_WAN_LOCAL=1 ./x/wan-c/wan-cli --ckpt-dir … --prompt "…" --out /tmp/t.mp4
```

## Speed / quality knobs

| Flag | Default | Notes |
|------|---------|-------|
| `--width` / `--height` | 480×832 | Any multiple of 16 |
| `--frames` | 81 | Must be `4n+1` |
| `--steps` | 50 | UniPC steps |
| `--cfg` | 5.0 | 0 disables CFG (faster, lower quality) |
| `--shift` | 5.0 | Flow sigma warp |
| `--solver` | unipc | or `dpmpp` |
| `--dtype` | f32 | or `f16` |

## Zerollama API

Set `ZEROLLAMA_WAN_CLI` to the `wan-cli` binary (or `backend_paths.wan_cli` in the video manifest). `POST /v1/videos` then runs `scripts/video/wan_c_generate.sh` instead of the Python Wan venv path. Python remains the fallback when `WAN_CLI` is unset.

## Broker ops

Host-CPU GRAPH ops registered in uma_toolkit (see `docs/WISHLIST_WAN_VIDEO.md`):

- `GEMM_F16` / `GEMV_F16`
- `LAYERNORM_MUL`
- `AFFINE_MUL_ADD` / `MODULATE6`
- `GROUP_NORM`

Metal acceleration and `ROPE3` / `CONV2D` / `CONV3D` wire-up continue on the UMA side; wan-c already names those ops in recipes.

## Parity

```bash
# Export tokenizer
python3 x/wan-c/tools/export_umt5_spm.py $CKPT/google/umt5-xxl/spiece.model -o umt5.vocab

# Dump Python intermediates (needs Wan venv)
python3 x/wan-c/tools/parity_dump.py --prompt "a cat" --out /tmp/wan_dumps
```
