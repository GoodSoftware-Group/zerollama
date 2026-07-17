package modality

import (
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// LogViTEmbedCacheSizing warns when the latest user turn has more image frames than
// the auto-grow cap (OLLAMA_IMAGE_EMBED_CACHE_MAX) and ViT radix is off.
//
// WHY: without a byte-budget radix pool, llamarunner stops growing at MAX and re-encodes
// overflow frames. With OLLAMA_VIT_RADIX (default), the content pool grows under the
// byte budget — skip the undersized warning.
func LogViTEmbedCacheSizing(msgs []api.Message) {
	if envconfig.EffectiveImageEmbedCacheBytes() > 0 {
		return
	}
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
		"hint", "raise OLLAMA_IMAGE_EMBED_CACHE_MAX or enable OLLAMA_VIT_RADIX for larger video clips",
	}
	if len(msg.VideoSpans) > 0 {
		attrs = append(attrs, "has_video_spans", true)
	}
	slog.Info("vision embed cache may be undersized for this turn", attrs...)
}
