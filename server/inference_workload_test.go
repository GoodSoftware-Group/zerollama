package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestInferenceWorkloadStatusRuntimeQueue(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"waiting":          2,
			"running":          1,
			"inference_state":  "running",
		})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	st := (&Server{}).inferenceWorkloadStatus(context.Background())
	if st.RuntimeWaiting != 2 || st.RuntimeRunning != 1 {
		t.Fatalf("runtime queue: waiting=%d running=%d", st.RuntimeWaiting, st.RuntimeRunning)
	}
	if !st.busy() {
		t.Fatal("expected busy with runtime queue")
	}
}

func TestCheckTrainingSubmitAllowedRuntimeLlamaLoaded(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"waiting": 0, "running": 0, "llama_server": true,
		})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	err := (&Server{}).checkTrainingSubmitAllowed(context.Background())
	if !errors.Is(err, ErrInferenceBacklogActive) {
		t.Fatalf("expected ErrInferenceBacklogActive, got %v", err)
	}
}

func TestCheckTrainingSubmitFailClosedOnHealthError(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED", "1")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:1")

	err := (&Server{}).checkTrainingSubmitAllowed(context.Background())
	if !errors.Is(err, ErrRuntimeHealthProbeFailed) {
		t.Fatalf("expected ErrRuntimeHealthProbeFailed, got %v", err)
	}
}

func TestCheckTrainingSubmitFailOpenOnHealthError(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED", "0")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:1")

	err := (&Server{}).checkTrainingSubmitAllowed(context.Background())
	if err != nil {
		t.Fatalf("expected fail-open when probe fails, got %v", err)
	}
}

func TestCheckTrainingSubmitAllowedRuntimeBusy(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"waiting": 1, "running": 0})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	err := (&Server{}).checkTrainingSubmitAllowed(context.Background())
	if err == nil {
		t.Fatal("expected error when runtime waiting > 0")
	}
}
