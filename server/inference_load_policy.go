package server

import (
	"context"

	"github.com/ollama/ollama/envconfig"
)

// runtimeBacklogPausesGgml reports whether the Python runtime queue should hold ggml loads.
func runtimeBacklogPausesGgml(ctx context.Context) bool {
	return runtimeBacklogPausesGgmlFrom(runtimeInferenceHealth(ctx))
}

func runtimeBacklogPausesGgmlFrom(h runtimeHealthSnapshot) bool {
	if !envconfig.GgmlPauseWhenRuntimeBusy() {
		return false
	}
	if !h.ok {
		return false
	}
	backlog := h.waiting + h.running
	return backlog >= envconfig.GgmlPauseRuntimeMinBacklog()
}

// trainingPausesGgml reports whether training currently blocks new ggml loads.
func (s *Server) trainingPausesGgml(ctx context.Context) bool {
	if s == nil || !envconfig.BlockInferenceDuringTraining() {
		return false
	}
	return s.trainingOccupiesGPU(ctx)
}

func (s *Server) inferencePausesGgml(ctx context.Context, h runtimeHealthSnapshot) bool {
	return s.trainingPausesGgml(ctx) || runtimeBacklogPausesGgmlFrom(h)
}

// updateSchedInferencePauses pauses ggml scheduling when training or runtime backlog requires it.
func (s *Server) updateSchedInferencePauses(ctx context.Context) {
	s.updateSchedInferencePausesFromHealth(ctx, runtimeInferenceHealth(ctx))
}

func (s *Server) updateSchedInferencePausesFromHealth(ctx context.Context, h runtimeHealthSnapshot) {
	if s == nil || s.sched == nil {
		return
	}
	if s.inferencePausesGgml(ctx, h) {
		s.sched.PauseNewLoads()
		return
	}
	s.sched.ResumeLoads()
}
