package server

import (
	"context"

	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
)

func (s *Server) statusResponse(ctx context.Context) api.StatusResponse {
	disabled, source := internalcloud.Status()
	return api.StatusResponse{
		Cloud: api.CloudStatus{
			Disabled: disabled,
			Source:   source,
		},
		Inference: s.inferenceStatus(ctx),
	}
}

func (s *Server) inferenceStatus(ctx context.Context) api.InferenceStatus {
	ggml := api.GgmlStatus{}
	if s != nil && s.sched != nil {
		snap := s.sched.InferenceFleetSnapshot()
		ggml.Pending = snap.Pending
		ggml.Active = snap.Active
		ggml.Loaded = snap.Loaded
		ggml.LoadsPaused = snap.LoadsPaused
		ggml.Loading = snap.Loading
		ggml.LoadedModels = snap.LoadedModels
		ggml.LoadedModelDetails = snap.LoadedModelDetails
	}

	training := &api.TrainingStatus{
		QueuePolicy: trainingQueuePolicy(s),
	}
	return api.InferenceStatus{
		Ggml:     ggml,
		Runtime:  runtimeStatusFromHealth(ctx, runtimeHealthProbeRequired()),
		Backend:  inferenceBackendPolicy(),
		Training: training,
	}
}

func runtimeStatusFromHealth(ctx context.Context, enabled bool) api.RuntimeStatus {
	runtime := api.RuntimeStatus{Enabled: enabled}
	if !enabled {
		return runtime
	}
	h := runtimeInferenceHealth(ctx)
	runtime.Available = h.ok
	if !h.ok {
		return runtime
	}
	runtime.Waiting = intPtr(h.waiting)
	runtime.Running = intPtr(h.running)
	runtime.LlamaLoaded = boolPtr(h.llamaLoaded)
	runtime.State = h.state
	return runtime
}

func intPtr(n int) *int       { return &n }
func boolPtr(b bool) *bool    { return &b }
