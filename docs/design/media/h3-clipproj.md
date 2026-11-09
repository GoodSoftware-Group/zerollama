# ClipProj — small Qwen3-VL → MiniMax-H3 conditioning

**Upstream:** [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) (MIT matrices) + [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj).  
**video-c host:** `x/video-c/family_h3/h3_clipproj_host.c` (`test_h3_clipproj_host`).

## Why it matters on Darwin

H3’s stock TE is Qwen3-VL-**32B** truncated to 50 layers (~15.7 GB NVFP4 on CUDA). ClipProj is a learned map so a **4B/8B** Qwen3-VL hidden state becomes the same `[seq, 5120]` conditioning tensor the DiT expects — no change to DiT / VAEs / sampler.

This does **not** replace downloading DiT+VAE weights; it only shrinks the TE side once those exist.

## Formula (node 0.1.4+)

```
xn   = (h - mean_in) / std_in
yn   = xn @ W                 # W is [d_in, 5120]
yn  += mlp(xn)                # optional residual: d_in → 16384 → 5120 + GELU
cond = yn * std_out + mean_out
cond[0] = sink_out            # attention-sink token (excluded from ridge fit)
```

| Encoder | `d_in` | Recommended file |
|---------|--------|------------------|
| Qwen3-VL-4B | 2560 | `mmh3-4b-ClipProj-celeb-mlp` (~304 MB) |
| Qwen3-VL-8B | 4096 | `mmh3-8b-ClipProj-celeb-mlp` (~386 MB) |

Controls (`control-zero`, `control-identity`) prove the matrix is doing work — run them before trusting a learned file.

## Local cache (this Mac)

```text
~/.zerollama/third_party/h3/clipproj/
  mmh3-ClipProj-control-zero.safetensors           # ~50 MB
  mmh3-ClipProj-control-identity.safetensors       # ~50 MB
  mmh3-4b-ClipProj-celeb-mlp.safetensors           # ~304 MB — learned + residual
```

`test_h3_clipproj_host` rematches control sink behaviour and the celeb-mlp residual
(`mlp.0` → GELU → `mlp.2`, fp16 weights decoded to f32).

**Qwen3-VL-4B-Instruct** (this Mac, ~8.3 GiB):

```text
~/.zerollama/models/Qwen3-VL-4B-Instruct/
  config.json          # hidden 2560, 36 layers, GQA 32/8, head_dim 128
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
```

`text_config` matches ClipProj-4B (`d_in=2560`). Tied embeddings (no `lm_head` shard).
mRoPE is `mrope_interleaved` with `mrope_section: [24, 20, 20]` (not the 32B `%3` table).
video-c `--embed` / `--generate` run `h3_qwen_te_4b_forward` for **24 decoder layers
and no final RMSNorm** (ClipProj calibration tap), then ClipProj. Full last-hidden
is `H3_QWEN_TE_LAYERS=36 H3_QWEN_TE_FINAL_NORM=1`. Override dir with `H3_QWEN_TE_DIR`.

Pull more with:

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
import os
d = os.path.expanduser("~/.zerollama/third_party/h3/clipproj")
for n in ["mmh3-4b-ClipProj-celeb-mlp.safetensors"]:  # or 8b
    print(hf_hub_download("NicoLab28/ClipProj-MiniMax-H3", n, local_dir=d))
PY
```

## Licence note

Matrices are MIT; they are derived from model activations — read MiniMax H3’s custom licence and Qwen Apache-2.0 before commercial use. Not affiliated with MiniMax / Alibaba / Comfy Org.
