# T11 / T11c — lite data recipes (ingest → `/api/train`)

**ROADMAP:** [GPU training T11](./ROADMAP.md#phases-training-track)  
**Status (Oct 2026):** **Partial** — CSV/TSV/JSONL/JSON/TXT stdlib ingest; PDF/DOCX optional; chat-export presets (ShareGPT / OpenAI / Alpaca); Python CLI + **`zerollama train recipe`**; optional **`POST /api/train/recipes`** upload helper (ingest only). **Not** Studio’s node editor.

## Why this exists

| Temptation | Why we do not |
|------------|----------------|
| Ship Unsloth Studio / a visual node editor | AGPL product surface; not our inference-first daemon |
| Require PDF OCR / layout ML | Optional path only; bad OCR must not block CSV/JSONL |
| Auto-POST to `/api/train/jobs` from CLI or `/recipes` | Operators choose host, priority, T6 window — ingest stays offline |
| Reimplement recipes in Go | One Python engine (`training_recipes.py`); Go is a thin PATH-friendly wrapper / HTTP shell |

**Product shape:** spreadsheet / chat export / short docs → `{prompt,response}` rows that drop into the existing SFT payload. Same train queue; no new backend.

## Shape

| Piece | Detail |
|-------|--------|
| Module | [`training_recipes.py`](../training_recipes.py) |
| Python CLI | `python training_recipes.py INPUT -o out.jsonl [--preset …]` |
| Go CLI | `zerollama train recipe INPUT -o out.jsonl` — [`cmd/train.go`](../cmd/train.go) |
| HTTP (T11c) | `POST /api/train/recipes` — JSON or multipart → `{n_rows, training_data, …}` (no job) |
| Smoke | [`scripts/training/t11_recipes_smoke.sh`](../scripts/training/t11_recipes_smoke.sh) |

**Why Go wraps Python:** operators already have `zerollama` on PATH after install; resolving `.venv-training` / `TRAINING_PYTHON` / `ZEROLLAMA_REPO` matches doctor and serve.

## Formats

| Format | Deps | Notes |
|--------|------|-------|
| CSV / TSV | stdlib | Columns `prompt`/`response` or aliases (`question`/`answer`, …) |
| JSONL / JSON | stdlib | prompt/response **or** chat-export presets below |
| TXT | stdlib | Blank-line blocks; first line prompt + rest response, or `---` separator |
| PDF | `pypdf` | Excerpt → summarize-style pairs (self-distill); clear `ImportError` if missing |
| DOCX | `python-docx` | Same chunk strategy as PDF |

### Chat-export presets (T11c)

| `--preset` / `preset` | Shape |
|-----------------------|--------|
| `auto` (default) | Detect ShareGPT → OpenAI messages → Alpaca → prompt/response |
| `sharegpt` | `conversations: [{from: human\|gpt, value}]` (aliases: vicuna) |
| `openai` | `messages: [{role, content}]` (aliases: chatgpt, chat) |
| `alpaca` | `instruction` + optional `input` → prompt; `output` → response |
| `prompt_response` | Force `prompt`/`response` (or question/answer) only |

**Why PDF “summarize” pairs:** a PDF alone has no gold assistant text. Chunk→“summarize this excerpt” / response=chunk is a honest self-distill default; operators replace rows offline when they have real targets.

## Example

```bash
# Either Python or Go (same recipe engine):
python training_recipes.py data.csv -o /tmp/sft.jsonl
zerollama train recipe data.csv -o /tmp/sft.jsonl

zerollama train recipe export.json --preset sharegpt -o /tmp/sft.jsonl
zerollama train recipe /tmp/sft.jsonl -o /tmp/train_payload.json \
  --model-name Qwen/Qwen2.5-0.5B-Instruct

# HTTP ingest (serve with training routes; never starts a GPU job):
curl -sS http://127.0.0.1:11435/api/train/recipes \
  -H 'Content-Type: application/json' \
  -d '{"content":"[{\"instruction\":\"Add\",\"input\":\"2+2\",\"output\":\"4\"}]",
       "preset":"alpaca","model_name":"demo"}'

# Merge training_data into POST /api/train/jobs:
# { "kind": "train", "payload": { ...contents of train_payload.json... } }
```

## Non-goals

- Visual node editor / Studio UI
- Guaranteeing PDF OCR quality
- Auto-submit to `/api/train/jobs` (operators POST the JSON)
- Full ChatGPT.com `mapping` tree exports (use OpenAI `messages` / ShareGPT dumps)

## Exit (T11c)

1. Presets ShareGPT / OpenAI / Alpaca covered by unit tests
2. `POST /api/train/recipes` returns `{n_rows, training_data}` without enqueue
3. Smoke: `./scripts/training/t11_recipes_smoke.sh`
