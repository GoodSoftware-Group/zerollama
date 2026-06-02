package vram

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

)

type mockEvictor struct {
	mu    sync.Mutex
	order []string
}

func (m *mockEvictor) PauseNewLoads() {
	m.mu.Lock()
	m.order = append(m.order, "pause")
	m.mu.Unlock()
}

func (m *mockEvictor) ResumeLoads() {
	m.mu.Lock()
	m.order = append(m.order, "resume")
	m.mu.Unlock()
}

func (m *mockEvictor) UnloadAllRunners() {
	m.mu.Lock()
	m.order = append(m.order, "unload")
	m.mu.Unlock()
}

func (m *mockEvictor) steps() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, len(m.order))
	copy(out, m.order)
	return out
}

func TestPrepareForTrainingOrder(t *testing.T) {
	ev := &mockEvictor{}
	PrepareForTraining(context.Background(), ev)
	want := []string{"pause", "unload"}
	got := ev.steps()
	if len(got) != len(want) {
		t.Fatalf("steps %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("steps %v want %v", got, want)
		}
	}
}

func TestPrepareForRuntimeInferenceOrder(t *testing.T) {
	var handoff, resume bool
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/internal/training-handoff":
			handoff = true
		case "/internal/inference/resume":
			resume = true
		default:
			http.NotFound(w, r)
		}
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	ev := &mockEvictor{}
	PrepareForRuntimeInference(context.Background(), ev)
	if got := ev.steps(); len(got) != 1 || got[0] != "unload" {
		t.Fatalf("evictor steps %v", got)
	}
	if !resume {
		t.Fatal("expected resume request")
	}
	_ = handoff
}

func TestPrepareForLegacyRunnerHandoff(t *testing.T) {
	var handoff bool
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/internal/training-handoff" {
			handoff = true
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	PrepareForLegacyRunner(context.Background())
	if !handoff {
		t.Fatal("expected training-handoff request")
	}
}
