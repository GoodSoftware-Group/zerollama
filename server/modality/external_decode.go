package modality

import (
	"context"

	"github.com/ollama/ollama/api"
)

// ExternalVideoDecodeHook, if non-nil, replaces in-process ffmpeg sampling for the native path.
// It runs after empty and max-size checks (same invariants as the ffmpeg path).
//
// Why optional: some deployments need one custom decode step (e.g. Python) without proxying
// full chat to another server—see docs/ROADMAP.md Phase E. Ollama still owns timeouts and caps.
var ExternalVideoDecodeHook func(ctx context.Context, policy VideoSamplingPolicy, video []byte) ([]api.ImageData, error)
