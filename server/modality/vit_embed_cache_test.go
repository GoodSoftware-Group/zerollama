package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLogViTEmbedCacheSizing_noPanic(t *testing.T) {
	t.Parallel()
	LogViTEmbedCacheSizing(nil)
	LogViTEmbedCacheSizing([]api.Message{
		{Role: "user", Images: make([]api.ImageData, 8)},
	})
	LogViTEmbedCacheSizing([]api.Message{
		{Role: "user", Images: make([]api.ImageData, 100)},
	})
}
