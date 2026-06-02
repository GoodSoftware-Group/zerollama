package server

import (
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// Scheduler debug logging (enable with OLLAMA_DEBUG=1). Filter logs: grep 'scheduler:'.

func schedPendingAttrs(pending *LlmRequest) []any {
	if pending == nil {
		return []any{"pending", nil}
	}
	modelName := ""
	modelKey := ""
	if pending.model != nil {
		modelName = pending.model.ShortName
		modelKey = schedulerModelKey(pending.model)
	}
	attrs := []any{
		"model", modelName,
		"model_key", modelKey,
		"fifo_seq", pending.fifoSeq,
		"sched_attempt", pending.schedAttempts,
		"num_ctx", pending.opts.NumCtx,
		"num_gpu", pending.opts.NumGPU,
		"num_predict", pending.opts.NumPredict,
	}
	if pending.sessionDuration != nil {
		attrs = append(attrs, "keep_alive", pending.sessionDuration.Duration.String())
	}
	if err := pending.ctx.Err(); err != nil {
		attrs = append(attrs, "ctx_err", err.Error())
	}
	return attrs
}

func schedRunnerAttrs(runner *runnerRef) []any {
	if runner == nil {
		return []any{"runner", nil}
	}
	runner.refMu.Lock()
	rc := runner.refCount
	loading := runner.loading
	runner.refMu.Unlock()
	name := ""
	if runner.model != nil {
		name = runner.model.ShortName
	}
	return []any{
		"runner_model", name,
		"runner_key", runner.modelKey,
		"ref_count", rc,
		"loading", loading,
		"pid", runner.pid,
		"vram", runner.vramSize,
		"session_duration", runner.sessionDuration.String(),
	}
}

func (s *Scheduler) schedLoadedModelNames() []string {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	names := make([]string, 0, len(s.loaded))
	for _, r := range s.loaded {
		if r.model != nil && r.model.ShortName != "" {
			names = append(names, r.model.ShortName)
		} else {
			names = append(names, r.modelKey)
		}
	}
	return names
}

func (s *Scheduler) schedActiveLoadingPath() string {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	if s.activeLoading == nil {
		return ""
	}
	return s.activeLoading.ModelPath()
}

func (s *Scheduler) schedSnapshot() []any {
	s.loadedMu.Lock()
	active := ""
	if s.activeLoading != nil {
		active = s.activeLoading.ModelPath()
	}
	s.loadedMu.Unlock()
	return []any{
		"pending_queue", s.pending.Len(),
		"loaded_models", strings.Join(s.schedLoadedModelNames(), ","),
		"active_loading", active,
		"loads_paused", s.loadsPaused.Load(),
		"loading_fifo_seq", s.loadingFifoSeq.Load(),
	}
}

func schedLogDebug(msg string, pending *LlmRequest, extra ...any) {
	attrs := append([]any{}, schedPendingAttrs(pending)...)
	attrs = append(attrs, extra...)
	slog.Debug("scheduler: "+msg, attrs...)
}

func schedLogInfo(msg string, pending *LlmRequest, extra ...any) {
	attrs := append([]any{}, schedPendingAttrs(pending)...)
	attrs = append(attrs, extra...)
	slog.Info("scheduler: "+msg, attrs...)
}

func schedLogWarn(msg string, pending *LlmRequest, extra ...any) {
	attrs := append([]any{}, schedPendingAttrs(pending)...)
	attrs = append(attrs, extra...)
	slog.Warn("scheduler: "+msg, attrs...)
}

func schedLogLoadPhase(pending *LlmRequest, phase string, start time.Time, extra ...any) {
	attrs := []any{"phase", phase, "elapsed", time.Since(start).Round(time.Millisecond)}
	attrs = append(attrs, extra...)
	schedLogDebug("load phase", pending, attrs...)
}

func schedKeepAliveDesc(d *api.Duration) string {
	if d == nil {
		return "default"
	}
	return d.Duration.String()
}
