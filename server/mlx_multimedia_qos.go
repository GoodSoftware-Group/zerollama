package server

import (
	"context"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/modality"
)

// MLX modality hints for server-inferred QoS (GET /api/version zerollama.qos.modalities).
const (
	mlxModalityText                = "text"
	mlxModalityVision              = "vision"
	mlxModalityVideoUnderstanding  = "video_understanding"
	mlxModalityImageGeneration     = "image_generation"
	mlxModalityVideoGeneration     = "video_generation"
)

const gpuMediaModelKey = "gpu:zerollama:media"

func mlxModalityFromChat(req *api.ChatRequest) string {
	if req == nil {
		return mlxModalityText
	}
	if modality.ChatRequestHasVideoPayload(req) {
		return mlxModalityVideoUnderstanding
	}
	if modality.ChatRequestHasMultimodalPayload(req) {
		return mlxModalityVision
	}
	return mlxModalityText
}

// ensureQoSDefaults injects modality-appropriate zerollama hints when the client omitted explicit QoS.
func ensureQoSDefaults(opts map[string]any, hints mlxScheduleHints) {
	if opts == nil {
		return
	}
	qos := mlxQoSFromOptions(opts)
	if qos.Explicit {
		return
	}
	z, _ := opts["zerollama"].(map[string]any)
	if z == nil {
		z = map[string]any{}
	}
	switch hints.Modality {
	case mlxModalityImageGeneration, mlxModalityVideoGeneration:
		if _, ok := z["qos_class"]; !ok {
			z["qos_class"] = "background"
		}
		if _, ok := z["cache_scope"]; !ok {
			z["cache_scope"] = qosCacheScopeShared
		}
	}
	if len(z) > 0 {
		opts["zerollama"] = z
	}
}

// waitRequestQoS blocks until MLX session policy allows this request. MLX models use per-runner
// gates; non-MLX GPU work (Wan video, external imagegen) waits behind any hot interactive MLX slot.
func (s *Server) waitRequestQoS(ctx context.Context, m *Model, opts map[string]any, hints mlxScheduleHints) error {
	if opts == nil {
		opts = map[string]any{}
	}
	ensureQoSDefaults(opts, hints)
	if m != nil && m.IsMLX() {
		return s.waitMLXSessionIdle(ctx, m, opts)
	}
	return s.waitBehindInteractiveMLX(ctx, opts, hints)
}

func (s *Server) waitBehindInteractiveMLX(ctx context.Context, opts map[string]any, hints mlxScheduleHints) error {
	if s == nil || s.sched == nil {
		return nil
	}
	rawKey := mlxSessionKey(opts)
	class, qos := classifyMLXSession(rawKey, hints, opts)
	sessionKey := injectMLXSessionKey(gpuMediaModelKey, rawKey, class, qos)
	return s.sched.mlxGate.waitBehindAnyInteractive(ctx, class, sessionKey)
}
