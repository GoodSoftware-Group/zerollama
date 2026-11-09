package slurm

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// WriteBundle writes payload.json and job.sh under dir. The script calls the
// same training.py entry the embed uses (process_training_request / GRPO /
// run_local_script) and exits. It does not start zerollama serve.
//
// dir must sit on a filesystem the compute node can read. Slurm spool on a
// login-node /tmp is invisible to the batch step.
func WriteBundle(dir, repo, python, kind string, payload []byte) (scriptPath string, err error) {
	if strings.TrimSpace(repo) == "" {
		return "", fmt.Errorf("%w: set ZEROLLAMA_REPO (directory containing training.py)", ErrBadRequest)
	}
	if python == "" {
		python = "python3"
	}
	if err := ValidatePython(python); err != nil {
		return "", err
	}
	switch kind {
	case "train", "grpo", "run_script":
	default:
		return "", fmt.Errorf("%w: kind %q", ErrBadRequest, kind)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	payloadPath := filepath.Join(dir, "payload.json")
	if len(payload) == 0 {
		payload = []byte("{}")
	}
	// 0600: a run_script payload can carry environment values, and the default
	// spool is under the system temp dir. The batch step runs as this user.
	if err := os.WriteFile(payloadPath, payload, 0o600); err != nil {
		return "", err
	}
	scriptPath = filepath.Join(dir, "job.sh")
	body := renderScript(repo, python, kind, payloadPath)
	if err := os.WriteFile(scriptPath, []byte(body), 0o700); err != nil {
		return "", err
	}
	return scriptPath, nil
}

func renderScript(repo, python, kind, payloadPath string) string {
	return fmt.Sprintf(`#!/bin/bash
# zerollama slurm queue client — training runs on the compute node, not the submit host.
set -euo pipefail
export ZEROLLAMA_REPO=%s
export ZEROLLAMA_SLURM_KIND=%s
export ZEROLLAMA_SLURM_PAYLOAD=%s
export PYTHONPATH="${ZEROLLAMA_REPO}${PYTHONPATH:+:$PYTHONPATH}"
cd "$ZEROLLAMA_REPO"
exec %s - <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["ZEROLLAMA_REPO"])
import training
with open(os.environ["ZEROLLAMA_SLURM_PAYLOAD"], encoding="utf-8") as f:
    data = json.load(f)
kind = os.environ.get("ZEROLLAMA_SLURM_KIND", "train")
if kind == "grpo":
    from training_grpo import process_grpo_request
    result = process_grpo_request(data)
elif kind == "run_script":
    result = training.run_local_script(data)
else:
    result = training.process_training_request(data)
print(json.dumps(result), flush=True)
ok = isinstance(result, dict) and result.get("status") == "ok"
sys.exit(0 if ok else 1)
PY
`, shellQuote(repo), shellQuote(kind), shellQuote(payloadPath), python)
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'"
}
