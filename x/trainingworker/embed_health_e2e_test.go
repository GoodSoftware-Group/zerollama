package trainingworker

// T4 (CPU CI smoke, real embed): unlike embed_health_test.go-style route smokes that accept
// 502 as "OK in CI" (no embedded Python running), this test actually calls Start() and
// requires a genuine healthy response. It only runs when a CPU-only (or GPU) training venv
// is provisioned and OLLAMA_TRAINING_PYTHONPATH/ZEROLLAMA_REPO points at training.py — see
// scripts/training/training_uv_venv.sh (TRAINING_UV_CPU_ONLY=1 for a CI-friendly CPU torch
// install) and docs/gpu-training.md.
//
// WHY a separate opt-in test rather than changing the existing CI-friendly smoke
// (server/training_ci_smoke_test.go): that test intentionally covers the "no embed available"
// path (route wiring only, no CGO Python init) and must keep passing on hosts with no venv at
// all. This test instead proves the *other* half of T4: that with a real ABI-matched venv, the
// embedded interpreter reaches "initialized" and reports healthy — not just "does not crash".

import (
	"context"
	"encoding/json"
	"os"
	"testing"
)

// TestEmbeddedTrainingHealthReal only runs when RUN_E2E_TRAINING_EMBED=1. It calls Start()
// for real (CGO Py_Initialize + import training, which imports torch/transformers/datasets/peft)
// and asserts TrainingHealthJSON returns status "ok" — the actual T4 exit condition, not the
// weaker "502 is acceptable" CI-only check.
func TestEmbeddedTrainingHealthReal(t *testing.T) {
	if os.Getenv("RUN_E2E_TRAINING_EMBED") != "1" {
		t.Skip("set RUN_E2E_TRAINING_EMBED=1 (with a provisioned .venv-training) to run")
	}

	client, err := Start(context.Background(), nil)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() {
		_ = client.trainUnload(context.Background())
	})

	raw, err := client.TrainingHealthJSON(context.Background())
	if err != nil {
		t.Fatalf("TrainingHealthJSON: %v", err)
	}

	var health struct {
		Status     string `json:"status"`
		ExtrasJSON string `json:"extrasJson"`
	}
	if err := json.Unmarshal(raw, &health); err != nil {
		t.Fatalf("unmarshal health JSON %q: %v", raw, err)
	}
	if health.Status != "ok" {
		t.Fatalf("expected status=ok, got %q (raw=%s)", health.Status, raw)
	}
}
