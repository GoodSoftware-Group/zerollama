package server

import (
	"context"
	"testing"
)

func TestCheckTrainingSubmitAllowedIgnoresGgmlLoadedWhenOptOut(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_GGML_LOADED", "0")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.loadedMu.Lock()
	sched.loaded["m"] = &runnerRef{modelKey: "m"}
	sched.loadedMu.Unlock()

	if err := (&Server{sched: sched}).checkTrainingSubmitAllowed(context.Background()); err != nil {
		t.Fatalf("expected allow when ggml-loaded wait disabled: %v", err)
	}
}
