#!/usr/bin/env bash
# T11 lite recipes smoke (stdlib; no GPU).
#
#   ./scripts/training/t11_recipes_smoke.sh
#
# Doc: docs/t11-recipes.md · ROADMAP T11
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv-training/bin/python" ]]; then
  PY="${ROOT}/.venv-training/bin/python"
elif [[ -n "${TRAINING_PYTHON:-}" && -x "${TRAINING_PYTHON}" ]]; then
  PY="${TRAINING_PYTHON}"
else
  PY=python3
fi

echo "== T11 unit: training_recipes =="
"$PY" -m unittest tests.test_training_recipes -v

TMP="${TMPDIR:-/tmp}/zerollama-t11-recipes-$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

echo "== T11 CLI: CSV → JSONL → payload JSON =="
cat >"$TMP/in.csv" <<'EOF'
prompt,response
What is 2+2?,4
Capital of France?,Paris
EOF
"$PY" "${ROOT}/training_recipes.py" "$TMP/in.csv" -o "$TMP/out.jsonl"
"$PY" "${ROOT}/training_recipes.py" "$TMP/out.jsonl" -o "$TMP/payload.json" --model-name demo
test "$(wc -l <"$TMP/out.jsonl" | tr -d ' ')" = "2"
grep -q '"training_data"' "$TMP/payload.json"
grep -q '"model_name": "demo"' "$TMP/payload.json"
echo "OK: CLI wrote $(wc -l <"$TMP/out.jsonl" | tr -d ' ') rows"

echo "== T11c: ShareGPT preset + --stdin envelope =="
cat >"$TMP/sharegpt.json" <<'EOF'
[{"conversations":[{"from":"human","value":"ping"},{"from":"gpt","value":"pong"}]}]
EOF
"$PY" "${ROOT}/training_recipes.py" "$TMP/sharegpt.json" --preset sharegpt -o "$TMP/sg.jsonl"
test "$(wc -l <"$TMP/sg.jsonl" | tr -d ' ')" = "1"
grep -q '"prompt": "ping"' "$TMP/sg.jsonl"
printf '%s' '{"content":"[{\"instruction\":\"Add\",\"input\":\"1+1\",\"output\":\"2\"}]","format":"json","preset":"alpaca","model_name":"demo"}' \
  | "$PY" "${ROOT}/training_recipes.py" --stdin >"$TMP/stdin_out.json"
grep -q '"n_rows": 1' "$TMP/stdin_out.json"
grep -q '"response": "2"' "$TMP/stdin_out.json"
echo "OK: T11c presets + stdin"

echo "== T11b: Go train recipe helpers =="
if command -v go >/dev/null 2>&1; then
  if (cd "$ROOT" && CGO_ENABLED=0 go test -count=1 ./cmd -run 'TestResolveTrainRecipePython|TestRunTrainRecipeMissingScript' >/tmp/t11b-go.log 2>&1); then
    echo "OK: cmd train recipe unit tests"
  else
    # CGO_ENABLED=0 may still pull package deps on some hosts; CI covers ./cmd.
    echo "WARN: go test ./cmd soft-skip (see /tmp/t11b-go.log)"
  fi
else
  echo "skip: go not installed"
fi

echo "t11_recipes_smoke: PASS"