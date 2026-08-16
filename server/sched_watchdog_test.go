package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func TestFindLRUIdleRunner(t *testing.T) {
	s := InitScheduler(t.Context())
	old := time.Now().Add(-time.Hour)
	newer := time.Now().Add(-time.Minute)

	s.loaded["a"] = &runnerRef{
		modelKey:   "a",
		modelPath:  "a",
		refCount:   0,
		lastUsedAt: old,
		llama:      &mockLlm{},
		loadDone:   make(chan struct{}),
	}
	close(s.loaded["a"].loadDone)

	s.loaded["b"] = &runnerRef{
		modelKey:   "b",
		modelPath:  "b",
		refCount:   0,
		lastUsedAt: newer,
		llama:      &mockLlm{},
		loadDone:   make(chan struct{}),
	}
	close(s.loaded["b"].loadDone)

	s.loaded["c"] = &runnerRef{
		modelKey:   "c",
		modelPath:  "c",
		refCount:   1,
		lastUsedAt: old.Add(-time.Hour),
		loadDone:   make(chan struct{}),
	}
	close(s.loaded["c"].loadDone)

	victim := s.findLRUIdleRunner()
	if victim == nil || victim.modelPath != "a" {
		t.Fatalf("victim=%v want a", victim)
	}
}

func TestWatchdogReclaimMemoryEvictsLRU(t *testing.T) {
	t.Setenv("ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD", "0.5")
	if envconfig.MemoryReclaimThreshold() != 0.5 {
		t.Fatalf("threshold not set")
	}

	ctx := t.Context()
	s := InitScheduler(ctx)
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 10 * format.GigaByte
		g.FreeMemory = 2 * format.GigaByte // 80% used
		return []ml.DeviceInfo{g}
	}

	old := time.Now().Add(-time.Hour)
	runner := &runnerRef{
		model:      &Model{ModelPath: "victim.gguf"},
		modelKey:   "victim.gguf",
		modelPath:  "victim.gguf",
		refCount:   0,
		lastUsedAt: old,
		llama:      &mockLlm{},
		loadDone:   make(chan struct{}),
		Options:    &api.Options{},
	}
	close(runner.loadDone)
	s.loaded[runner.modelKey] = runner

	s.watchdogReclaimMemory(ctx)

	select {
	case <-s.expiredCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for eviction")
	}
}

func TestWatchdogReclaimShrinksBeforeEvict(t *testing.T) {
	t.Setenv("ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD", "0.5")
	ctx := t.Context()
	s := InitScheduler(ctx)
	calls := 0
	s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		calls++
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "CUDA"}}
		g.TotalMemory = 10 * format.GigaByte
		if calls == 1 {
			g.FreeMemory = 2 * format.GigaByte // 80% used
		} else {
			g.FreeMemory = 6 * format.GigaByte // shrink freed KV
		}
		return []ml.DeviceInfo{g}
	}

	llm := &shrinkMockLlm{mockLlm: mockLlm{contextLength: 32768}}
	runner := &runnerRef{
		model:      &Model{ModelPath: "qwen.gguf"},
		modelKey:   "qwen.gguf",
		modelPath:  "qwen.gguf",
		refCount:   0,
		lastUsedAt: time.Now().Add(-time.Hour),
		llama:      llm,
		loadDone:   make(chan struct{}),
		Options:    &api.Options{Runner: api.Runner{NumCtx: 32768}},
	}
	close(runner.loadDone)
	s.loaded[runner.modelKey] = runner

	s.watchdogReclaimMemory(ctx)

	select {
	case <-s.expiredCh:
		t.Fatal("should not evict after idle kv shrink dropped usage under threshold")
	case <-time.After(50 * time.Millisecond):
	}
	if llm.calls != 1 {
		t.Fatalf("shrink calls=%d want 1", llm.calls)
	}
}

func TestWaitUntilReadyWaitsForLoad(t *testing.T) {
	runner := &runnerRef{
		loading:  true,
		loadDone: make(chan struct{}),
		llama:    &mockLlm{},
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- runner.waitUntilReady(ctx)
	}()

	time.Sleep(50 * time.Millisecond)
	runner.refMu.Lock()
	runner.loading = false
	runner.refMu.Unlock()
	close(runner.loadDone)

	if err := <-done; err != nil {
		t.Fatalf("waitUntilReady: %v", err)
	}
}
