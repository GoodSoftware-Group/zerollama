package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestFinalizeInferenceCoordinationClearsMirror(t *testing.T) {
	var mu sync.Mutex
	var last map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/go-coordination" {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		last = body
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)

	schedCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(schedCtx)
	s := &Server{sched: sched}
	sched.PauseNewLoads()

	s.finalizeInferenceCoordination(context.Background())

	deadline := time.Now().Add(2 * time.Second)
	for sched.loadsPaused.Load() {
		if time.Now().After(deadline) {
			t.Fatal("scheduler still paused after finalize")
		}
		time.Sleep(10 * time.Millisecond)
	}

	mu.Lock()
	defer mu.Unlock()
	if last == nil {
		t.Fatal("expected go-coordination push on finalize")
	}
	if last["ggml_loads_paused"] != false {
		t.Fatalf("ggml_loads_paused=%v", last["ggml_loads_paused"])
	}
	if last["training_gpu_blocked"] != false {
		t.Fatalf("training_gpu_blocked=%v", last["training_gpu_blocked"])
	}
}
