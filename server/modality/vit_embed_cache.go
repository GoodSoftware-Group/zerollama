package modality

import (
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// LogViTEmbedCacheSizing warns when the latest user turn has more image frames than
// the auto-grow cap (OLLAMA_IMAGE_EMBED_CACHE_MAX).
//
// WHY: llamarunner auto-grows the embed LRU up to the max; turns above the cap still
// re-encode evicted frames — operators grep this to raise OLLAMA_IMAGE_EMBED_CACHE_MAX.
func LogViTEmbedCacheSizing(msgs []api.Message) {
	maxSlots := envconfig.ImageEmbedCacheMax()
	idx := lastUserMessageIndex(msgs)
	if idx < 0 {
		return
	}
	msg := msgs[idx]
	frames := len(msg.Images)
	if frames <= maxSlots {
		return
	}
	attrs := []any{
		"frames", frames,
		"cache_max", maxSlots,
		"hint", "raise OLLAMA_IMAGE_EMBED_CACHE_MAX for larger video clips",
	}
	if len(msg.VideoSpans) > 0 {
		attrs = append(attrs, "has_video_spans", true)
	}
	slog.Info("vision embed cache may be undersized for this turn", attrs...)
}
