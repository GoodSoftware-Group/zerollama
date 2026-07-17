# Native FP8 GGUF weights (E4M3 / E5M2)

**Status (Jul 2026):** Built on ggml-org pin `86d86ed4` — patches **0076–0079**, packaged `libggml-cuda` probe **PASS**.

## Why this exists

Hugging Face FP8 checkpoints (ModelOpt, DeepSeek-style `weight_scale_inv`, `float8_e4m3fn` / `float8_e5m2`) used to land in GGUF only after **full dequant to F16/BF16** (or re-quant to Q4_K/Q8_0). That:

1. **Inflates disk and VRAM** during convert and load.
2. **Throws away the native E4M3/E5M2 payload** already in the safetensors.
3. Blocks a clean CUDA path for “load FP8 GGUF → matmul without a mandatory F16 expand.”

Zerollama therefore adds **block FP8 weight types** in ggml, keeps convert optional via `--fp8-native`, and wires CUDA **convert / get_rows / MMVQ / MMQ**.

This is **weight FP8**, not **FP8 KV cache**. Fork KV remains QJL / Polar / TBQ (see [gpu-profiles-l2.md](./gpu-profiles-l2.md)).

## Type layout

| ggml type | ID | Block | Bytes | IEEE payload |
|-----------|----|-------|-------|--------------|
| `GGML_TYPE_FP8_E4M3` | **51** | F16 `d` + 32× E4M3FN | 34 | OCP / PyTorch `float8_e4m3fn` (max finite 448; NaN = `S.1111.111`) |
| `GGML_TYPE_FP8_E5M2` | **52** | F16 `d` + 32× E5M2 | 34 | IEEE `float8_e5m2` (max finite 57344; Inf/NaN in exp=31) |

**Why Q8_0-shaped (QK=32, 34 B):** Reuse existing CUDA MMQ tile geometry and CPU vec_dot pairing with `Q8_0` activations. Not NVFP4 (type 40) and not Blackwell NVFP4 MMA.

**Why IDs 51/52:** Types 39–50 are MXFP4/NVFP4 + eliza QJL/Polar/TBQ + E8_2@43. E8_2 stays at **43** in this tree (not 51 — older changelog notes about E8_2@51 referred to a prior in-tree dig).

Llama ftypes: `LLAMA_FTYPE_MOSTLY_FP8_E4M3 = 42`, `MOSTLY_FP8_E5M2 = 43`.

## Patches

| Patch | What | Why |
|-------|------|-----|
| **0073** | Type 51 + CPU quant/dot + CUDA convert/get_rows + gguf-py + `--fp8-native` (ModelOpt / scalar–row) | Minimal loadable FP8 GGUF |
| **0074** | CUDA MMVQ (float dequant × Q8_1) + MMQ (per-block amax → int8, reuse Q8_0 DP4A/MMA) | Avoid dequant-all-then-GEMM on every decode |
| **0075** | Expand 128×128 `weight_scale_inv` onto GGML 32-wide blocks | DeepSeek/HF block scales without full F16 expand when `block_size[-1] % 32 == 0` |
| **0076** | Type 52 E5M2 twin (CPU + CUDA + convert) | Same path for `torch.float8_e5m2` checkpoints |

Vendor tip: see `LLAMA_CPP_VENDOR_HEAD`. Apply via `./scripts/apply_llama_vendor_patches.sh`.

## Convert

```bash
# From vendor (or synced) tree:
python3 convert_hf_to_gguf.py /path/to/hf-fp8 --fp8-native --outfile model-fp8.gguf
```

**Why `--fp8-native`:** Default convert still dequants FP8 → F16 for maximum compatibility. Native pack keeps E4M3/E5M2 bytes and stores the scale(s) as F16 block `d`.

Supported scale shapes:

- Scalar / per-row (ModelOpt `.weight_scale`)
- compressed-tensors `float-quantized` with `strategy: tensor|channel` (per-tensor / per-row `.weight_scale`)
- 2D block maps (e.g. `weight_block_size=[128,128]`) when the **last** block dim is a multiple of 32

Otherwise convert logs a warning and falls back to dequant.

## CUDA path

| Op | Behavior |
|----|----------|
| **convert / get_rows** | Soft or `__nv_fp8_*` dequant to F16/F32/BF16 |
| **MMVQ** | Per-element E4M3/E5M2 → float × Q8_1 activation |
| **MMQ** | Requant each 32-block to int8 via amax, then existing Q8_0 tiles |

**Why MMQ requant:** sm_89 has no native FP8 tensor-core path in this stack; int8 DP4A/MMA is the practical matmul. Correctness first; Blackwell FP8/NVFP4 MMA remains a separate P2 lane.

**Host build:** Prefer `./scripts/build_llama_server_container.sh` (CUDA 12.8 devel). Host CUDA 13.x + system GCC often fails `__cudaLaunch` arity mismatches.

Install: `sudo ./scripts/install_cuda_llama_server.sh` then restart `zerollama-runtime`.

## Probes & smokes

```bash
./scripts/fp8_cuda_probe.sh          # packaged/vendor libggml-cuda markers (no GGUF)
PYTHONPATH=vendor/llama-cpp-86d86ed4/gguf-py python3 scripts/fp8_e4m3_gguf_roundtrip.py
PYTHONPATH=vendor/llama-cpp-86d86ed4/gguf-py python3 scripts/fp8_e5m2_gguf_roundtrip.py
FP8_GGUF=/path/to/fp8.gguf ./scripts/fp8_cuda_load_smoke.sh   # load + /completion
```

**Host fixture (dual-4090, Jul 2026):** [nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e](https://huggingface.co/nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e) → `convert_hf_to_gguf.py --fp8-native` → `/mnt/ssd2/models/fp8/tinyllama-fp8-e2e/tinyllama-1.1b-fp8_e4m3.gguf` (~1.3 GiB; 154× `FP8_E4M3`). Load smoke on GPU1: `llama-server -c 2048 -ngl 99` → `/health` OK, `/completion` ~477 tok/s, ~1.6 GiB VRAM. Artifact: `/tmp/fp8-cuda-load-smoke.json`.

Runtime: `/health.llama_patches.cuda_weight_formats` → `{fp8_e4m3, fp8_e5m2, nvfp4, mxfp4, libggml_cuda}` when the serving runtime tree includes the Jul 2026 probe code. Binary probe is authoritative if `/opt` runtime is older.

## Operator notes

- **Prod fork:** Keep `llama_fork: stock` unless evaluating L2 TBQ/QJL profiles.
- **Not MLX mxfp8:** `x/create` / imagegen FP8 is Apple/MLX-only ([cuda-lanes.md](./cuda-lanes.md)).
- **VRAM estimates:** `runtime/gguf_estimate.py` layout `(32, 34)` for types 51 and 52.

## Related

- [cuda-lanes.md](./cuda-lanes.md) — quantization roadmap table
- [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) — pin + patch series
- [gpu-profiles-l2.md](./gpu-profiles-l2.md) — fork KV (orthogonal to weight FP8)
