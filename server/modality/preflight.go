package modality

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

// PreflightVideoVisionBudget returns an error when an upper bound on vision tokens for messages
// that include video clearly exceeds numCtx (cheap check before ffmpeg runs).
//
// Why only messages with Videos[]: counting every historical image would reject many valid
// multi-turn chats (truncate may drop old turns). We only estimate the turn(s) about to incur
// new video expansion plus stills on those same messages—enough to catch obviously oversized
// requests without duplicating full prompt truncation logic.
//
// Why vision-only: text tokenization is expensive here; the goal is a fast fail before temp files
// and subprocess. Users may still hit context limits from text; that path remains truncate/shift.
func PreflightVideoVisionBudget(policy VideoSamplingPolicy, numCtx int, req *api.ChatRequest) error {
	if numCtx <= 0 || req == nil {
		return nil
	}
	tp := policy.TokensPerImage
	if tp <= 0 {
		tp = 768
	}
	var total int64
	for _, msg := range req.Messages {
		if len(msg.Videos) == 0 {
			continue
		}
		total += int64(len(msg.Images)) * int64(tp)
		total += int64(len(msg.Videos)) * int64(policy.MaxFrames) * int64(tp)
	}
	if total > int64(numCtx) {
		return fmt.Errorf("estimated vision tokens (~%d, upper bound for messages with video: still images + max expanded frames) exceed num_ctx (%d); reduce frames, fewer videos, raise num_ctx, or lower OLLAMA_VIDEO_MAX_FRAMES / manifest max_frames", total, numCtx)
	}
	return nil
}
