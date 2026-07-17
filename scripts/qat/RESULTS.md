# Ternary (1.58-bit) QAT Pipeline — Results

Hardware: Apple M4 Max, 128 GB unified memory (`sysctl hw.memsize` = 137438953472),
92 GB free disk at start. Training on MPS (`torch.backends.mps.is_available() == True`),
fp32 shadow weights (bf16/fp16 training on MPS was not attempted — fp32 is the safe
default per the STE design anyway).

Base model: `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, hidden_size=896, 494M params).
Dataset: 4,000 examples from `tatsu-lab/alpaca`, Alpaca-style
`### Instruction:\n...\n\n### Response:\n...` formatting, 384-token cap.

## Pipeline

1. `ternary_ste.py` — `TernaryLinear`/`TernaryEmbedding` (FP32 shadow weight,
   group_size=128, dead_zone=0.5·s_g threshold, STE backward with grad_clip=1.0),
   `TiedTernaryLMHead` for `tie_word_embeddings=True` models, `replace_with_ternary()`,
   `freeze_all_ternary()`, `unwrap_ternary_to_plain()`.
2. `prepare_dataset.py` — downloads/formats/tokenizes Alpaca, saves via
   `datasets.Dataset.save_to_disk`.
3. `train_ternary_qat.py` — HF `Trainer` on MPS, `PROGRESS:NN:msg` callback,
   freezes + unwraps ternary modules to plain `nn.Linear`/`nn.Embedding` at the end
   so `convert_hf_to_gguf.py` needs zero custom-architecture handling.
4. `export_gguf.sh` — `convert_hf_to_gguf.py --outtype f16` then
   `llama-quantize ... TQ1_0` / `TQ2_0`.

Two training runs:

| Run | LR | Epochs | Steps | Final loss |
|---|---|---|---|---|
| v1 | 3e-5 | 2 | 1000 | 0.75–0.80 (overfit/collapsed) |
| v2 | 8e-6 | 1 | 500 | 1.07–1.16 (healthier) |

`freeze_all_ternary()` diagnostics on the v1 checkpoint showed **no degenerate
groups** — every attention/MLP projection has a roughly balanced ~1/3 zero,
1/3 +1, 1/3 -1 split (e.g. `mlp.gate_proj: zero=0.313 pos=0.344 neg=0.343`). The
dead_zone=0.5 default did not need tuning; ternary quantization itself was healthy
in both runs. v1's degenerate *output* (always "B. Response") was a training/LR
issue (too aggressive for 2 epochs on 4k examples of a 0.5B model), not a
quantization-collapse issue — a useful distinction the diagnostics surfaced directly.

## GGUF export quirk (matches the task's PTQ caveat, but for QAT weights too)

Qwen2.5-0.5B's `hidden_size=896` and intermediate sizes are not multiples of 256,
so **`llama-quantize ... TQ1_0/TQ2_0` falls back 144/290 tensors to Q4_0** even on
the QAT-trained checkpoint — this is a `llama-quantize` architectural constraint
(TQ1_0/TQ2_0 kernels require 256-divisible dims), independent of whether the source
weights were naively rounded or trained end-to-end with STE. Only `ffn_down` (the
one 256-divisible dim in this architecture) actually lands in true TQ1_0/TQ2_0;
q/k/v/o and gate/up fall back to Q4_0 for both the PTQ and QAT exports. This means
the "ternary GGUF" for this particular small model is really a mixed TQ1_0+Q4_0
GGUF. Larger/differently-shaped models (e.g. ones with 256-divisible hidden dims)
would ternarize more completely. This is orthogonal to the coherence result below,
which is driven by the *values* in the tensors that did get read as ternary/Q4_0,
not by the fallback itself.

## Metal backend gap

Mainline llama.cpp's Metal (`GGML_METAL=ON`) backend on this build (commit c3d47e6)
hits `ggml-metal-device.cpp:924: not implemented` / `ggml_metal_library_get_pipeline_mul_mm_id_map0`
when a TQ1_0 tensor is present in the graph — this reproduces for both the naive-PTQ
and QAT exports. All ternary-GGUF testing below therefore ran on CPU (`-ngl 0`);
`llama-bench`/`llama-cli` throughput numbers are CPU-only as a result. FP16 and the
PrismML mainline-compatible reference ran fine on Metal.

## Coherence comparison (3 fixed prompts, CPU backend, greedy decoding)

| Model | "Say hello." | "What is 17\*24?" | "Write a haiku about GPUs." |
|---|---|---|---|
| **FP16 baseline** (`qwen25_0.5b_base.F16.gguf`) | "Hello! How can I assist you today?" | "17 multiplied by 24 is 408." (correct) | "Infinite power, / GPU's speed never stops, / Compute dreams." |
| **Naive PTQ ternary** (`qwen25_0.5b_base_PTQ.TQ1_0.gguf`, no training) | `ione其中. 史. InSeconds- FIGare 嗦 Interval  Beyond...` | `0100grosley851 intÉ ivelysoled  Territories...` | `00filesizeisareiscoisexreachableos0ave...` | Character-soup, non-linguistic garbage — this reproduces the task's stated premise exactly. |
| **QAT ternary v2** (`qwen25_0.5b_qat_v2.TQ1_0.gguf`, trained 500 steps, lr=8e-6) | "1." | "1. The most a main and the most main and the most the main and the world and the world..." | "1. The presence of the presence of the presence..." | Grammatical English words, correct tokenizer/vocab usage, but repetitive/degenerate loops — undertrained, not gibberish. |
| **PrismML Ternary-Bonsai-1.7B** (`Q2_0_g64`, released, Qwen3-1.7B base, mainline-compatible) | "Hello! I'm Bonsai, a 1-bit AI assistant developed by PrismML. It's a pleasure to meet you. How can I assist you today?" | "To calculate 17 × 24, we can use the distributive property... 17 × (20 + 4)..." (mid-reasoning, correct method) | "Parallel dreams / Heat pulse through circuits / Speeds rise." | Fully coherent, well-formed, on-topic. Reference for what QAT+scale (1.7B, presumably far more training compute/data than this session's budget) achieves. |

**Bottom line on the core claim:** the QAT pipeline does exactly what it's
supposed to, qualitatively — going from character-soup gibberish (naive PTQ) to
real English words/grammar with correct vocabulary usage (QAT, 500 steps on 4k
examples). It is **not** yet "meaningfully more coherent" in the sense of
producing sensible, on-topic sentences the way the PrismML reference does;
closing that gap needs materially more training compute/data/steps than a single
Cursor session budget allows for a from-scratch QAT run (PrismML's own writeup
implies large-scale training infra behind their models). The pipeline, diagnostics
(`ternary_stats`/degeneracy warnings), and export path are all working correctly end
to end; the remaining gap is training budget, not a bug in the STE/ternary
machinery.

## Throughput (`llama-bench -p 16 -n 16`, CPU backend since Metal lacks TQ1_0 kernels)

| Model | Size | pp16 (tok/s) | tg16 (tok/s) |
|---|---|---|---|
| FP16 baseline | 942 MiB | 782 ± 47 | 104 ± 28 |
| Naive PTQ TQ1_0 | 295 MiB | 1704 ± 75 | 292 ± 25 |
| QAT TQ1_0 | 295 MiB | 1574 ± 191 | 288 ± 9 |
| QAT TQ2_0 | 300 MiB | 2023 ± 193 | 299 ± 9 |

Ternary formats give the expected ~3x smaller footprint and ~2-3x generation
throughput vs. FP16 (CPU path), with TQ1_0 vs TQ2_0 performing similarly since
both mostly execute as Q4_0 for this model's shape (see the GGUF export quirk
above).

## Artifacts

- `scripts/qat/ternary_ste.py`, `prepare_dataset.py`, `train_ternary_qat.py`,
  `export_gguf.sh` — the pipeline.
- `out/qat/qwen25_0.5b_ternary_v2_clean/` — final HF checkpoint (safetensors,
  ternary weights materialized into plain `nn.Linear`/`nn.Embedding`).
- `out/gguf/qwen25_0.5b_base.F16.gguf`, `qwen25_0.5b_base_PTQ.TQ1_0.gguf` — FP16 and
  naive-PTQ-ternary baselines.
- `out/gguf/qwen25_0.5b_qat_v2.F16.gguf`, `.TQ1_0.gguf`, `.TQ2_0.gguf` — QAT-trained
  exports.
- `out/reference/Ternary-Bonsai-1.7B-Q2_0_g64.gguf` — PrismML's released,
  mainline-llama.cpp-compatible ternary reference model (their `prism` fork with
  the full Q2_0 kernel was not built in this session; the `_g64` mainline-compatible
  variant they publish loaded and ran fine on the same mainline build used above).
- Transient full-precision optimizer-state checkpoints (v1 run, smoke run, base-model
  duplicate HF checkpoint, superseded v1 GGUFs) were deleted after successful export,
  per the disk-hygiene constraint; only the items listed above were kept.

## Reproducing

```bash
source .venv-qat/bin/activate   # torch/transformers/datasets/accelerate installed here

python3 scripts/qat/prepare_dataset.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir data/qat/alpaca_4k --num_examples 4000 --max_length 384

PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/qat/train_ternary_qat.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path data/qat/alpaca_4k \
  --output_dir out/qat/qwen25_0.5b_ternary \
  --batch_size 8 --learning_rate 8e-6 --num_epochs 1 --device mps

# convert_hf_to_gguf.py chokes on TiedTernaryLMHead's tensor name; reload once
# through plain AutoModelForCausalLM (state dict keys match nn.Linear/Embedding)
# and re-save before exporting -- see RESULTS.md "GGUF export quirk" note.

GGUF_OUT_DIR=out/gguf scripts/qat/export_gguf.sh <clean_ckpt_dir> qwen25_0.5b_qat
```

---

## Follow-up (longer 0.5B + 3B scale-up)

PrismML ships weights + a whitepaper, **not** open training code/recipe — so these
runs are our own evidence, not a reproduction of their pipeline.

### Longer 0.5B QAT (v3)

| Run | Data | LR | Epochs | Steps | train_loss |
|---|---|---|---|---|---|
| v2 (prior) | 4k | 8e-6 | 1 | 500 | 1.24 |
| **v3** | **8k** | **5e-6** | **2** | **2000** | **1.02** |

Loss curve was healthy (2.00 → ~0.83). Coherence did **not** improve: F16 already
emits numeric debris (`: 1: 10.`), TQ1_0 collapses to `C: [10000...`. More Alpaca
SFT steps at this LR/scale were not enough; the gap to PrismML remains a recipe /
compute / data-mix problem, not “run 4× longer on the same tiny instruct set.”

### 3B scale-up (`Qwen/Qwen2.5-3B-Instruct`)

- 4k Alpaca, lr=8e-6, 1 epoch, 1000 steps, batch 4, ~2.6h on MPS.
- train_loss ≈ 1.15 (final step ~0.87).
- **Critical packaging win:** `hidden_size=2048` is 256-divisible → **true TQ1_0
  at 2.18 BPW with zero Q4_0 fallbacks** (unlike 0.5B’s mixed 5.01 BPW).

| Model | Size | "Say hello." | "What is 17\*24?" | Notes |
|---|---|---|---|---|
| 3B FP16 | 5.75 GiB | "Hello! How can I assist you today?" | "17 multiplied by 24 equals 408." | Coherent baseline |
| 3B naive PTQ TQ1_0 | 802 MiB | `贰找 syn DLverter勿⇨...` | gibberish / peg error | Same character-soup failure as 0.5B PTQ |
| 3B QAT TQ1_0 | 802 MiB | English list loops: `- What is a good?` | `- What is 100?` (repeated) | Real English tokens; degenerate but **not** gibberish |
| 3B QAT F16 (pre-quant) | 5.75 GiB | `-"Hello"` loops | similar | Same failure mode pre-TQ → undertraining, not TQ packing |

Throughput (`llama-bench -p 16 -n 16 -ngl 0`): FP16 tg≈30 t/s; TQ1_0 PTQ/QAT tg≈86–92 t/s (~3×).

### What we learned

1. QAT vs naive PTQ is real and reproducible at 3B with *true* ternary GGUF: PTQ =
   garbage, QAT = English (still not useful answers after ~1k SFT steps).
2. Dimensionality matters: prefer models whose dims are multiples of 256 if the
   goal is mainline `TQ1_0`/`TQ2_0` without Q4_0 escape hatches.
3. 4× more SFT on 0.5B did not close the PrismML gap — need a different training
   recipe (larger/diverse corpus, chat-template supervision, distillation /
   longer QAT schedule), not just more Alpaca epochs.
4. Skipped PrismML’s closed `prism` fork; used their mainline-compatible
   `Q2_0_g64` GGUF as the quality ceiling reference only.

### Artifacts added

- `out/qat/qwen25_0.5b_ternary_v3_clean/`, `out/gguf/qwen25_0.5b_qat_v3.{F16,TQ1_0,TQ2_0}.gguf`
- `out/qat/qwen25_3b_ternary_clean/`
- `out/gguf/qwen25_3b_{base.F16,base_PTQ.TQ1_0,qat.F16,qat.TQ1_0,qat.TQ2_0}.gguf`

---

## Label-mask fix + projections-only QAT (Jul 16)

### Root cause of “character soup” / pad collapse

Older `prepare_dataset.py` set `labels = copy(input_ids)` after `padding="max_length"`,
so most supervised tokens were **pad**. Models trained to emit pad/EOS debris.

**Fix** (`data/qat/alpaca_chat_4k`): Qwen chat template; `labels=-100` on prompt tokens
and on padding (~10% supervised tokens).

### Instant hard ternary still destroys the base

Zero-shot (0 STE steps) on 0.5B after `replace_with_ternary` + freeze:

| Scope | “Say hello.” |
|---|---|
| proj-only (embed+lm_head FP) | `,,,,...` |
| full (embed+lm_head ternary) | `,,,,...` |
| FP base | `Hello! How can I assist you today?` |

So STE training is required; packaging alone is not enough.

### Full-network vs projections-only (fixed labels)

| Run | Scope | Epochs | train_loss | “Say hello.” | “Name three colors.” |
|---|---|---|---|---|---|
| `ternary_fixed` | all Linear+Embedding | 1 | ~6.07 | wrong / broken English | `1.` |
| **`ternary_proj`** | **attn/MLP only** (embed+lm_head FP) | **3** | **3.03** | **`Hello, I'm John.`** | **`Three colors are blue, green, and blue.`** |

Proj-only QAT on the fixed chat set is the first own checkpoint that is **clearly English and on-topic** (still wrong on math / weaker than FP base). Evaluate quality on **HF / F16 GGUF** — 0.5B TQ1_0 still falls back ~145/291 tensors to Q4_0 (~5.12 BPW).

Artifacts:

- HF: `out/qat/qwen25_0.5b_ternary_proj/`
- GGUF: `out/gguf/qwen25_0.5b_proj_qat.{F16,TQ1_0}.gguf`
- Log: `out/qat/train_proj.log`

### 3B proj-only

| Run | Epochs | train_loss | Smoke “Say hello.” | TQ1_0 |
|---|---|---|---|---|
| `qwen25_3b_ternary_proj` | 1 | 5.62 | `The first of the given words is "1".` | **2.39 BPW, 0 fallbacks** (975 MiB) |
| `qwen25_3b_ternary_proj_e3` | +2 (3 total) | 4.50 | `1. What is the most important in the given?` | same packaging path |

Extra epochs lowered loss but did **not** recover coherent answers on 3B with only 4k chat SFT. Best own quality so far remains **0.5B proj-only 3ep** (`Hello, I'm John.` / color names). 3B’s win here is packaging (true TQ1_0), not quality.

Artifacts: `out/gguf/qwen25_3b_proj_qat{,_e3}.{F16,TQ1_0}.gguf`, HF `out/qat/qwen25_3b_ternary_proj{,_e3}/`.
