package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// trainingQueuePolicy reports configured T6 gates and live defer depth for GET /api/status.
func trainingQueuePolicy(s *Server) api.TrainingQueuePolicy {
	p := api.TrainingQueuePolicy{
		WaitInferenceIdle:        envconfig.TrainingWaitInferenceIdle(),
		WaitGgmlLoaded:           envconfig.TrainingWaitGgmlLoaded(),
		WaitFailClosed:           envconfig.TrainingWaitFailClosed(),
		QueueOnBusy:              envconfig.TrainingQueueOnBusy(),
		AllowedWindowEnabled:     envconfig.TrainingAllowedWindowEnabled(),
		AllowedWindowMisconfigured: envconfig.TrainingAllowedWindowMisconfigured(),
		CrossQueueFifo:           true,
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
	return p
}
