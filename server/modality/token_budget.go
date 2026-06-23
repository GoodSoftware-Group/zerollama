package modality

import (
	"github.com/ollama/ollama/api"
)

// MultimodalTokenEstimate is an upper-bound vision token budget by modality (preflight heuristic).
// Values flow to api.Metrics and OpenAI usage.prompt_tokens_details for observability.
// Why heuristic: exact vision-tower counts require model-specific processors; this catches
// obvious over-budget requests before ffmpeg and surfaces SGLang-shaped usage for agents.
type MultimodalTokenEstimate struct {
	ImageTokens int
	VideoTokens int
	AudioTokens int
}

func (e MultimodalTokenEstimate) HasValues() bool {
	return e.ImageTokens > 0 || e.VideoTokens > 0 || e.AudioTokens > 0
}

// EstimateMultimodalTokens counts still images vs expanded video frames using VideoSpans.
// Call after ExpandVideosInChatRequest so Videos are cleared and spans are populated.
//
// Scoping mirrors PreflightVideoVisionBudget post-expand behavior: only the latest user
// message contributes. Agents echo multimodal history every turn; counting old frames
// would inflate usage and access-log modality fields on follow-ups.
func EstimateMultimodalTokens(policy VideoSamplingPolicy, req *api.ChatRequest) MultimodalTokenEstimate {
	if req == nil {
		return MultimodalTokenEstimate{}
	}
	tp := policy.TokensPerImage
	if tp <= 0 {
		tp = 768
	}
	lastUser := lastUserMessageIndex(req.Messages)
	var out MultimodalTokenEstimate
	for i, msg := range req.Messages {
		if lastUser >= 0 && i != lastUser {
			continue
		}
		if n := preprocessedLayoutTokenCount(msg); n > 0 {
			assignPreprocessedUsage(msg, &out)
			out.AudioTokens += len(msg.AudioClips) * tp
			continue
		}
		videoFrames := 0
		videoTokens := 0
		for _, sp := range msg.VideoSpans {
			videoFrames += sp.FrameCount
			videoTokens += videoSpanVisionTokens(sp, tp, policy.visionSpatialMergeSize())
		}
		stillImages := len(msg.Images) - videoFrames
		if stillImages < 0 {
			stillImages = 0
		}
		out.ImageTokens += stillImages * tp
		if videoTokens > 0 {
			out.VideoTokens += videoTokens
		} else {
			out.VideoTokens += videoFrames * tp
		}
		out.AudioTokens += len(msg.AudioClips) * tp
	}
	return out
}
