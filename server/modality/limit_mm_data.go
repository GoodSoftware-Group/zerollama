package modality

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

// LimitMMDataPerRequest caps multimodal inputs on the latest user turn (SGLang
// limit_mm_data_per_request). Zero means no limit for that modality.
type LimitMMDataPerRequest struct {
	Image int
	Video int
	Audio int
}

func (l LimitMMDataPerRequest) enabled() bool {
	return l.Image > 0 || l.Video > 0 || l.Audio > 0
}

// latestUserMMCounts counts distinct MM inputs on the latest user message before
// ffmpeg expand. Pre-expanded spans count as video clips; still images exclude
// frame rasters attributed to video_spans.
func latestUserMMCounts(req *api.ChatRequest) (images, videos, audio int, ok bool) {
	if req == nil {
		return 0, 0, 0, false
	}
	idx := lastUserMessageIndex(req.Messages)
	if idx < 0 {
		return 0, 0, 0, false
	}
	msg := req.Messages[idx]
	videos = len(msg.Videos) + len(msg.VideoSpans)
	audio = len(msg.AudioClips)
	images = len(msg.Images)
	if len(msg.VideoSpans) > 0 {
		frameCount := 0
		for _, sp := range msg.VideoSpans {
			frameCount += sp.FrameCount
		}
		images -= frameCount
		if images < 0 {
			images = 0
		}
	}
	return images, videos, audio, true
}

// PreflightLimitMMDataPerRequest rejects requests that exceed configured per-modality
// caps on the latest user message. Why latest user only: agent history may echo
// prior clips; SGLang caps the active turn's MM attachments, not the full thread.
func PreflightLimitMMDataPerRequest(limits LimitMMDataPerRequest, req *api.ChatRequest) error {
	if !limits.enabled() {
		return nil
	}
	images, videos, audio, ok := latestUserMMCounts(req)
	if !ok {
		return nil
	}
	if limits.Image > 0 && images > limits.Image {
		return fmt.Errorf("too many images on latest user message (%d > limit %d); set OLLAMA_LIMIT_MM_DATA_PER_REQUEST or reduce attachments", images, limits.Image)
	}
	if limits.Video > 0 && videos > limits.Video {
		return fmt.Errorf("too many videos on latest user message (%d > limit %d); set OLLAMA_LIMIT_MM_DATA_PER_REQUEST or reduce clips", videos, limits.Video)
	}
	if limits.Audio > 0 && audio > limits.Audio {
		return fmt.Errorf("too many audio clips on latest user message (%d > limit %d); set OLLAMA_LIMIT_MM_DATA_PER_REQUEST or reduce attachments", audio, limits.Audio)
	}
	return nil
}
