// Scheduler watchdog: periodic LRU eviction under VRAM pressure and optional
// busy-timeout unload. Why: multi-model agent workloads need automatic reclaim
// without operators running `zerollama stop` on every stuck session (LocalAI
// WatchDog pattern, adapted to ggml runner lifecycle).
package server

import (
	"context"
	"log/slog"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

func (s *Scheduler) processSchedWatchdog(ctx context.Context) {
	interval := envconfig.SchedWatchdogInterval()
	if interval <= 0 {
		return
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.watchdogTick(ctx)
		}
	}
}

func (s *Scheduler) watchdogTick(ctx context.Context) {
	s.watchdogReclaimMemory(ctx)
	s.watchdogBusyRunners()
}

func (s *Scheduler) watchdogReclaimMemory(ctx context.Context) {
	threshold := envconfig.MemoryReclaimThreshold()
	if threshold <= 0 {
		return
	}

	runners := s.LoadedRunnersForDiscovery()
	if len(runners) == 0 {
		return
	}

	gpus := s.getGpuFn(ctx, runners)
	for _, gpu := range gpus {
		if gpu.TotalMemory == 0 {
			continue
		}
		used := 1.0 - float64(gpu.FreeMemory)/float64(gpu.TotalMemory)
		if used < threshold {
			continue
		}
		victim := s.findLRUIdleRunner()
		if victim == nil {
			return
		}
		slog.Info("watchdog evicting idle LRU runner for memory reclaim",
			"model", victim.modelPath,
			"gpu", gpu.ID,
			"vram_used_ratio", used,
			"threshold", threshold,
			"total_vram", format.HumanBytes2(gpu.TotalMemory),
			"free_vram", format.HumanBytes2(gpu.FreeMemory),
		)
		s.scheduleExpiredRunner(victim)
		return
	}
}

func (s *Scheduler) watchdogBusyRunners() {
	timeout := envconfig.RunnerBusyTimeout()
	if timeout <= 0 {
		return
	}

	now := time.Now()
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()

	for _, runner := range runners {
		runner.refMu.Lock()
		refCount := runner.refCount
		busySince := runner.busySince
		modelPath := runner.modelPath
		runner.refMu.Unlock()
		if refCount == 0 || busySince.IsZero() {
			continue
		}
		if now.Sub(busySince) <= timeout {
			continue
		}
		slog.Warn("watchdog evicting runner that exceeded busy timeout",
			"model", modelPath,
			"busy_for", now.Sub(busySince),
			"timeout", timeout,
		)
		s.scheduleExpiredRunner(runner)
	}
}

func (s *Scheduler) findLRUIdleRunner() *runnerRef {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()

	var victim *runnerRef
	var oldest time.Time
	for _, runner := range s.loaded {
		runner.refMu.Lock()
		idle := runner.refCount == 0 && !runner.loading && runner.llama != nil
		lastUsed := runner.lastUsedAt
		runner.refMu.Unlock()
		if !idle {
			continue
		}
		if victim == nil || lastUsed.Before(oldest) {
			victim = runner
			oldest = lastUsed
		}
	}
	return victim
}
