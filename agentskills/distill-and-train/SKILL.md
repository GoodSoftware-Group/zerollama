---
name: distill-and-train
description: "Fine-tune (LoRA/QLoRA) a local model on a zerollama server, including distilling a larger model into a smaller one via synthetic SFT data, plus the ternary QAT path for extreme quantization."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, training, fine-tuning, lora, qlora, distillation, qat, gpu]
    category: mlops
    related_skills: [zerollama-integration, batch-inference, fleet-vram-admission, download-model]
---

# Distill & Train Skill

Fine-tune a local model on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server via the embedded training worker (`POST /api/train/jobs`, LoRA/QLoRA
supervised fine-tuning). Zerollama has **no built-in teacher-logit KD loss**
— "distillation" here means the standard practical recipe: use a larger
model to **generate synthetic (prompt, response) training data**, then
**supervised-fine-tune** a smaller model on it. There is also a separate,
more advanced **ternary (1.58-bit) QAT** path for extreme quantization,
which is a standalone script, not a server job.

## When to Use

- The user wants to fine-tune a local model on custom data (SFT, style/persona
  adaptation, task specialization)
- The user wants to "distill" a big model's behavior into a smaller, cheaper
  model to run locally
- Producing a LoRA adapter to import via a Modelfile `ADAPTER` line
- Extreme compression (ternary/1.58-bit) of a small model for GGUF export

## Prerequisites

- `OLLAMA_TRAINING=true` (default when the embedded PyTorch worker is
  available) — training venv set up per `docs/gpu-training.md`
  (`.venv-training`, uv-managed, ABI-matched to the embedded `libpython3.X`)
- GPU (CUDA) for QLoRA; LoRA works on CUDA or Apple Silicon MPS (QLoRA is
  CUDA-only — `bitsandbytes`)
- For distillation: a larger "teacher" model already pulled/reachable
  (local or via `/v1/chat/completions/batch`) to generate training examples

## Workflow A — Distill a big model into a small one (SFT on synthetic data)

1. **Generate synthetic training data from the teacher model.** Use
   `batch-inference` (`POST /v1/chat/completions/batch`, max 8 per call) or
   sequential `/v1/chat/completions` calls against the larger model to
   produce `{"prompt": ..., "response": ...}` pairs covering the target
   task/style. Collect a few hundred to a few thousand pairs depending on
   task narrowness.

   ```bash
   curl -s http://localhost:11434/v1/chat/completions/batch \
     -H 'content-type: application/json' \
     -d '{
       "model": "qwen3-coder-next:6bit",
       "requests": [
         {"messages":[{"role":"user","content":"Explain recursion to a beginner."}]},
         {"messages":[{"role":"user","content":"Explain a hash map to a beginner."}]}
       ]
     }'
   # → collect {prompt, response} pairs from each choices[0].message.content
   ```

2. **Submit a fine-tuning job on the small (student) model** with that data:

   ```bash
   curl -s http://localhost:11434/api/train/jobs -d '{
     "kind": "train",
     "payload": {
       "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
       "output_dir": "/tmp/training_output/distill-explainer",
       "training_data": [
         {"prompt": "Explain recursion to a beginner.", "response": "Recursion is..."},
         {"prompt": "Explain a hash map to a beginner.", "response": "A hash map is..."}
       ],
       "num_epochs": 3,
       "batch_size": 4,
       "learning_rate": 0.0002,
       "use_lora": true,
       "use_qlora": false,
       "lora_rank": 16,
       "lora_alpha": 32
     },
     "priority": "normal"
   }'
   ```

3. **Poll status**, then **import the LoRA adapter**:

   ```bash
   curl -s http://localhost:11434/api/train/jobs/<job_id>
   # once completed, output_dir contains lora_adapter/ (adapter_model.safetensors, adapter_config.json, tokenizer files)
   ```

   ```bash
   # Modelfile
   FROM Qwen/Qwen2.5-0.5B-Instruct
   ADAPTER /tmp/training_output/distill-explainer/lora_adapter
   ```

   ```bash
   curl -s http://localhost:11434/api/create -d '{"model":"distill-explainer","files":{"Modelfile":"..."}}'
   ```

## Workflow B — Plain fine-tuning (no teacher model)

Same as above, minus step 1 — supply your own `training_data` directly
(existing dataset, logs, human-written examples).

## `train` job payload reference

| Field | Default | Notes |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | HF model id or local path (base to fine-tune) |
| `output_dir` | `/tmp/training_output` | Where checkpoints + `lora_adapter/` land |
| `training_data` | `[]` | List of `{"prompt": ..., "response": ...}` — formatted internally as `### Instruction:\n{prompt}\n\n### Response:\n{response}` |
| `num_epochs` | `3` | |
| `batch_size` | `4` | Per-device |
| `learning_rate` | `2e-4` | |
| `use_lora` | `true` | Recommended default; works on CUDA and Apple Silicon MPS |
| `use_qlora` | `false` | **CUDA only** — rejected with an explicit error on MPS |
| `lora_rank` | `16` | |
| `lora_alpha` | `32.0` | |

Wrap the payload in `{"kind": "train", "payload": {...}, "priority": "normal"|"low"|"high", "queue_on_busy": true}`.

## Job lifecycle API

| Endpoint | Method | Notes |
|---|---|---|
| `/api/train/jobs` | `POST` | Submit `{kind, payload, priority?, queue_on_busy?}` |
| `/api/train/jobs` | `GET` | List recent jobs |
| `/api/train/jobs/:id` | `GET` | Poll status/progress (embedded mode has **no push events** — poll this) |
| `/api/train/jobs/:id` | `DELETE` | Cancel a running Python job or a waiting `defer-*` job |
| `/api/train/unload` | `POST` | Unload the cached training model from GPU |
| `/api/train/status` | `GET` | Health + queue extras |

## Workflow C — Ternary (1.58-bit) QAT (advanced, standalone)

Not a server job — a standalone script pipeline for extreme quantization,
useful when you need a much smaller footprint than LoRA/int4 GGUF gives
you. Only tested on small models (~0.5B) so far.

```bash
python3 scripts/qat/prepare_dataset.py   # downloads/formats/tokenizes a dataset
python3 scripts/qat/train_ternary_qat.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --dataset_path data/qat/alpaca_qwen25_0.5b \
  --output_dir out/qat/qwen25_0.5b_ternary \
  --num_epochs 1 --batch_size 4 --learning_rate 2e-5
./scripts/qat/export_gguf.sh   # convert_hf_to_gguf.py --outtype f16, then llama-quantize ... TQ1_0/TQ2_0
```

Use a **low learning rate** (`~8e-6`) and few epochs — `scripts/qat/RESULTS.md`
documents a run at `3e-5`/2 epochs collapsing into a degenerate output
(always the same answer) while `8e-6`/1 epoch stayed healthy, even though
the ternary weight distribution itself was fine in both runs (roughly
balanced zero/+1/-1 splits) — the failure was an LR/epoch issue, not a
quantization issue.

## Pitfalls

- **No native teacher-logit distillation loss** — this is SFT-on-teacher-outputs
  ("distillation" in the practical/data sense), not KL-divergence-to-teacher-logits
  training. Don't expect a `teacher_model` field in the job payload.
- **QLoRA on Apple Silicon fails immediately** — `use_qlora: true` off-CUDA
  returns an explicit error; use `use_lora: true` / `use_qlora: false` on Mac.
- **VRAM contention with inference** — training and chat/image/video share
  the GPU; expect ggml/runtime models to be evicted while a job runs
  (`ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING`, on by default). Check
  `fleet-vram-admission` before submitting a big job on a busy host.
- **No push progress events** — embedded mode only supports polling
  `GET /api/train/jobs/:id`; don't wait for a webhook or SSE stream.
- **Mid-training OOM fails the job, no checkpoint resume** — the VRAM
  bridge frees memory for the *next* job but does not resume the failed
  one; reduce `batch_size`/`lora_rank` or retry rather than expecting
  automatic recovery.
- **`ADAPTER` import expects the exact `lora_adapter/` layout** — PEFT
  `adapter_model.safetensors` + `adapter_config.json` + tokenizer files;
  don't point `ADAPTER` at `output_dir` itself if the adapter is nested one
  level deeper.
- **Training venv ABI must match the embedded interpreter** — `No module
  named 'torch'` or `Failed to load PyTorch C extensions` almost always
  means `.venv-training`'s Python version doesn't match the linked
  `libpython3.X`; see `docs/gpu-training.md` troubleshooting table.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `batch-inference` — generating synthetic teacher data efficiently
- `fleet-vram-admission` — checking GPU headroom before submitting a training job
- `download-model` — pulling the base/teacher models this workflow needs
