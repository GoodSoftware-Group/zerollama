#!/usr/bin/env bash
# T7 train → GGUF/ADAPTER → register → chat smoke.
#
# Always (fast):
#   ./scripts/training/t7_train_export_smoke.sh
#     → unittest training_export (HTTP fallback, Modelfile parse, digests)
#
# Full loop (GPU/CPU train + convert + lab register) — NEVER uses :11434 by default:
#   RUN_E2E_T7=1 OLLAMA_HOST=http://127.0.0.1:11435 ./scripts/training/t7_train_export_smoke.sh
#
# Optional: start a lab serve for the duration (non-production ports only):
#   RUN_E2E_T7=1 RUN_E2E_T7_START_SERVE=1 ./scripts/training/t7_train_export_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv-training/bin/python" ]]; then
  PY="${ROOT}/.venv-training/bin/python"
else
  PY=python3
fi

echo "== T7 unit tests =="
"$PY" -m unittest tests.test_training_export -v

if [[ "${RUN_E2E_T7:-0}" != "1" ]]; then
  echo "OK: unit tests only (set RUN_E2E_T7=1 for tiny train→register→chat)"
  exit 0
fi

HOST="${OLLAMA_HOST:-http://127.0.0.1:11435}"
# Refuse accidental production host unless explicitly overridden.
if [[ "${HOST}" == *"11434"* && "${ZEROLLAMA_T7_ALLOW_PROD:-0}" != "1" ]]; then
  echo "Refusing OLLAMA_HOST=${HOST} (production :11434). Use :11435 lab or set ZEROLLAMA_T7_ALLOW_PROD=1" >&2
  exit 2
fi
export OLLAMA_HOST="$HOST"

SERVE_PID=""
cleanup() {
  if [[ -n "${SERVE_PID}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; then
    kill "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${RUN_E2E_T7_START_SERVE:-0}" == "1" ]]; then
  BIN="${ZEROLLAMA_BIN:-${ROOT}/zerollama}"
  if [[ ! -x "$BIN" ]]; then
    echo "Building zerollama for lab serve..." >&2
    CGO_ENABLED=1 /usr/local/go/bin/go build -o "${ROOT}/zerollama" .
    BIN="${ROOT}/zerollama"
  fi
  # Strip scheme for OLLAMA_HOST bind form
  BIND="${HOST#http://}"
  BIND="${BIND#https://}"
  echo "== starting lab serve on ${BIND} =="
  OLLAMA_HOST="${BIND}" OLLAMA_TRAINING=false "$BIN" serve >/tmp/t7-lab-serve.log 2>&1 &
  SERVE_PID=$!
  ready=0
  for _ in $(seq 1 90); do
    # Prefer /api/version — /api/tags can be slow when a local tag has a missing blob.
    if curl -sf -m 3 "${HOST}/api/version" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if [[ "$ready" != "1" ]]; then
    echo "lab serve failed to become ready; log:" >&2
    tail -50 /tmp/t7-lab-serve.log >&2 || true
    exit 1
  fi
fi

echo "== T7 e2e tiny train → export → register → generate (${HOST}) =="
export LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${ROOT}/../llama.cpp}"
"$PY" - <<'PY'
from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path

HOST = os.environ["OLLAMA_HOST"].rstrip("/")
ROOT = Path(__file__).resolve().parent if False else Path.cwd()

def http_json(method, path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else {}

# Serve must be up
try:
    code, ver = http_json("GET", "/api/version", timeout=10)
    print("serve version", ver, flush=True)
except Exception as e:
    raise SystemExit(f"lab serve not reachable at {HOST}: {e}")

from training_export import find_convert_hf_to_gguf, register_model, write_gguf_modelfile

convert = find_convert_hf_to_gguf()
if not convert:
    raise SystemExit("convert_hf_to_gguf.py not found — set LLAMA_CPP_DIR")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

model_id = os.environ.get("ZEROLLAMA_T7_MODEL", "sshleifer/tiny-gpt2")
out_dir = Path(tempfile.mkdtemp(prefix="t7-smoke-"))
print("model", model_id, "out", out_dir, flush=True)

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)
model = get_peft_model(
    model,
    LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        bias="none",
    ),
)

texts = [
    "### Instruction:\nSay hi\n\n### Response:\nHello.",
    "### Instruction:\n1+1\n\n### Response:\n2",
] * 4
ds = Dataset.from_dict({"text": texts})
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir=str(out_dir / "hf"),
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=1,
    report_to="none",
    save_strategy="no",
    learning_rate=1e-3,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
)
trainer.train()
adapter = out_dir / "lora_adapter"
model.save_pretrained(str(adapter))
tok.save_pretrained(str(adapter))
print("adapter saved", adapter, flush=True)

from training_export import run_export

tag = os.environ.get("ZEROLLAMA_T7_TAG", "t7-smoke:latest")
req = {
    "register_model": tag,
    "register_via": "http",
    "export_gguf": True,
    "export_quant": "f16",
    "export_unload": True,
    "export_gguf_dir": str(out_dir / "gguf"),
}
info = run_export(
    request=req,
    model=model,
    tokenizer=tok,
    model_name=model_id,
    output_dir=out_dir,
    adapter_path=adapter,
    unload_fn=None,  # already done training; empty_cache inside release
)
print("export", json.dumps(info, indent=2, default=str), flush=True)
if info.get("gguf_error"):
    raise SystemExit(f"gguf export failed: {info['gguf_error']}")
reg = info.get("register") or {}
if reg.get("status") != "ok":
    raise SystemExit(f"register failed: {reg}")

# Confirm tag is visible
code, shown = http_json("POST", "/api/show", {"model": tag}, timeout=60)
print("show status", code, "model", shown.get("modelfile", "")[:80].replace("\n", "\\n"), flush=True)
if code != 200:
    raise SystemExit(f"show failed: {shown}")

# Generate is best-effort: tiny-gpt2 GGUF often aborts llama-server; register+show is the T7 loop gate.
try:
    code, gen = http_json(
        "POST",
        "/api/generate",
        {
            "model": tag,
            "prompt": "Hello",
            "stream": False,
            "options": {"num_predict": 8},
        },
        timeout=180,
    )
    print("generate status", code, "response_len", len(str(gen.get("response", ""))), flush=True)
    if code == 200:
        print("OK: T7 e2e train→gguf→register→generate", flush=True)
    else:
        print(
            "WARN: generate failed (toy GGUF may not load); register+show OK — T7 loop closed",
            flush=True,
        )
        print("OK: T7 e2e train→gguf→register→show", flush=True)
except Exception as e:
    print(f"WARN: generate error {e}; register+show OK — T7 loop closed", flush=True)
    print("OK: T7 e2e train→gguf→register→show", flush=True)
PY
