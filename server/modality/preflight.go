package modality

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/api"
)

// PreflightVideoVisionBudget returns an error when an upper bound on vision tokens for messages
// that include raw video, pre-expanded video_spans, or audio clearly exceeds numCtx (cheap check
// before ffmpeg runs).
//
// Why only multimodal turns: counting every historical image would reject many valid multi-turn
// chats (truncate may drop old turns). Raw videos[] on any message are counted (about to expand).
// Pre-expanded spans and audio count only the latest user message so echoed history does not
// inflate the estimate.
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
	lastUser := lastUserMessageIndex(req.Messages)
	var total int64
	for i, msg := range req.Messages {
		if len(msg.Videos) == 0 && len(msg.VideoSpans) == 0 && len(msg.AudioClips) == 0 && len(msg.PaddedInputIDs) == 0 {
			continue
		}
		if len(msg.Videos) == 0 && lastUser >= 0 && i != lastUser {
			// Pre-expanded spans / audio / padded layout on historical turns — truncate may drop them.
			continue
		}
		if n := preprocessedLayoutTokenCount(msg); n > 0 {
			total += int64(n)
			total += int64(len(msg.AudioClips)) * int64(tp)
			continue
		}
		videoFrames := 0
		videoTokens := 0
		for _, sp := range msg.VideoSpans {
			videoFrames += sp.FrameCount
			videoTokens += videoSpanVisionTokens(sp, tp, policy.visionSpatialMergeSize())
		}
		if len(msg.Videos) > 0 {
			// About to expand: stills on this turn plus upper bound per raw clip.
			total += int64(len(msg.Images)) * int64(tp)
			total += int64(len(msg.Videos)) * int64(policy.MaxFrames) * int64(tp)
		} else {
			// Pre-expanded (SGLang-style): Images already hold sampled frames; VideoSpans
			// record how many came from each clip. grid_thw overrides flat frame×768 heuristic.
			still := len(msg.Images) - videoFrames
			if still < 0 {
				still = 0
			}
			total += int64(still) * int64(tp)
			if videoTokens > 0 {
				total += int64(videoTokens)
			} else {
				total += int64(videoFrames) * int64(tp)
			}
		}
		total += int64(len(msg.AudioClips)) * int64(tp)
	}
	if total > int64(numCtx) {
		return fmt.Errorf("estimated vision tokens (~%d, upper bound for messages with video: still images + max expanded frames) exceed num_ctx (%d); reduce frames, fewer videos, raise num_ctx, or lower OLLAMA_VIDEO_MAX_FRAMES / manifest max_frames", total, numCtx)
	}
	return nil
}

// PreflightMllamaSingleImage rejects video or multi-image turns before ffmpeg when the loaded
// model is mllama (Llama 3.2 Vision), which supports one raster per message.
//
// Why before expand: the same check exists post-expand in server/prompt.go; failing early saves
// temp files and subprocess work and names the fix (max_frames=1 or a multi-image VLM).
// Historical pre-expanded turns are skipped (truncate may drop them); multi-image history still
// fails because mllama cannot render those messages.
func PreflightMllamaSingleImage(policy VideoSamplingPolicy, modelFamilies []string, req *api.ChatRequest) error {
	if req == nil || !slices.Contains(modelFamilies, "mllama") {
		return nil
	}
	maxFrames := policy.MaxFrames
	if maxFrames <= 0 {
		maxFrames = 1
	}
	lastUser := lastUserMessageIndex(req.Messages)
	for i, msg := range req.Messages {
		count := len(msg.Images)
		hasRaw := len(msg.Videos) > 0
		hasSpans := len(msg.VideoSpans) > 0

		if !hasRaw && !hasSpans && count <= 1 {
			continue
		}
		if hasRaw {
			count += len(msg.Videos) * maxFrames
		} else if hasSpans && lastUser >= 0 && i != lastUser {
			continue
		}
		if count > 1 {
			return fmt.Errorf("this model only supports one image; video expands to multiple frames — use a single still image, set manifest max_frames or OLLAMA_VIDEO_MAX_FRAMES to 1, or choose a multi-image vision model")
		}
	}
	return nil
}
