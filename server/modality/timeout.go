package modality

import (
	"context"

	"github.com/ollama/ollama/envconfig"
)

// Subprocess contexts (override with OLLAMA_*_TIMEOUT as a Go duration string, e.g. "15m").
func whisperCtx(parent context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(parent, envconfig.ModalityWhisperTimeout())
}

func piperCtx(parent context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(parent, envconfig.ModalityPiperTimeout())
}

func externalImageCtx(parent context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(parent, envconfig.ModalityExternalImageTimeout())
}
