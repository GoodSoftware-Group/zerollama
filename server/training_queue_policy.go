package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// trainingQueuePolicy reports configured T6 gates and live defer depth for GET /api/status.
func trainingQueuePolicy(s *Server) api.TrainingQueuePolicy {
	p := api.TrainingQueuePolicy{
		WaitInferenceIdle:            envconfig.TrainingWaitInferenceIdle(),
		WaitGgmlLoaded:               envconfig.TrainingWaitGgmlLoaded(),
		WaitFailClosed:               envconfig.TrainingWaitFailClosed(),
		QueueOnBusy:                  envconfig.TrainingQueueOnBusy(),
		DeferPausesGgml:              envconfig.TrainingDeferPausesGgml(),
		BlockInferenceDuringTraining: envconfig.BlockInferenceDuringTraining(),
		AllowedWindowEnabled:         envconfig.TrainingAllowedWindowEnabled(),
		AllowedWindowMisconfigured:   envconfig.TrainingAllowedWindowMisconfigured(),
		CrossQueueFifo:               true,
	}
	if envconfig.TrainingAllowedWindowConfigured() {
		p.AllowedWindow = envconfig.TrainingAllowedWindowLabel()
	}
	if s != nil && s.trainingDefer != nil {
		st := s.trainingDefer.coordinationStats()
		if n, ok := st["defer_waiting"].(int); ok {
			p.DeferWaiting = n
		}
		if n, ok := st["defer_tracked"].(int); ok {
			p.DeferTracked = n
		}
	}
	if s != nil {
		// Best-effort live bits for fleet dashboards (no extra health probe here).
		p.TrainingGPUBusy = s.trainingVRAMBlocked || s.videoExclusiveActive()
		if s.sched != nil {
			p.GgmlLoadsPaused = s.sched.LoadsPaused()
		}
	}
	return p
}
