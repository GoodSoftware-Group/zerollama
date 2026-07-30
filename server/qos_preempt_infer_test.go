package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner"
)

func TestSoftPreemptBindCancelTake(t *testing.T) {
	t.Parallel()
	g := newMLXAgentGate()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g.bindPreemptCancel("model-a", "hermes:agent:1", cancel)
	g.mu.Lock()
	g.cancelSessionLocked("model-a", "hermes:agent:1", "lower_wait_interactive")
	g.mu.Unlock()

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("expected cancel to fire")
	}
	if got := g.takePreemptReason("model-a", "hermes:agent:1"); got != "lower_wait_interactive" {
		t.Fatalf("reason=%q", got)
	}
	if got := g.takePreemptReason("model-a", "hermes:agent:1"); got != "" {
		t.Fatalf("reason should clear, got %q", got)
	}
}

func TestMaybeEnqueueGeneratePreempted(t *testing.T) {
	t.Parallel()
	s := &Server{sched: InitScheduler(t.Context())}
	m := &Model{Digest: "d1", ModelPath: "/tmp/m.gguf"}
	key := "hermes:agent:gen"
	opts := map[string]any{"prompt_cache_key": key}

	ctx, cancel := context.WithCancel(context.Background())
	s.sched.mlxGate.bindPreemptCancel(schedulerModelKey(m), key, cancel)
	s.sched.mlxGate.mu.Lock()
	s.sched.mlxGate.cancelSessionLocked(schedulerModelKey(m), key, "lower_wait_interactive")
	s.sched.mlxGate.mu.Unlock()

	ch := make(chan any, 1)
	sent := false
	start := time.Now().Add(-2 * time.Second)
	loaded := time.Now().Add(-time.Second)
	if !s.maybeEnqueueGeneratePreempted(ch, m, opts, "m", "partial", &sent, start, loaded, nil) {
		t.Fatal("expected preempt handled")
	}
	if !sent {
		t.Fatal("sentDone")
	}
	msg := <-ch
	res, ok := msg.(api.GenerateResponse)
	if !ok {
		t.Fatalf("type %T", msg)
	}
	if !res.Done || res.DoneReason != "preempted" || res.PreemptedReason != "lower_wait_interactive" {
		t.Fatalf("%+v", res)
	}
	if res.Response != "partial" {
		t.Fatalf("partial=%q", res.Response)
	}
	_ = ctx
}

func TestCacheKeyPinnedHookRegistered(t *testing.T) {
	// server init sets mlxrunner.CacheKeyPinned; ensure it is non-nil after package load.
	if mlxrunner.CacheKeyPinned == nil {
		t.Fatal("CacheKeyPinned hook not registered")
	}
}
