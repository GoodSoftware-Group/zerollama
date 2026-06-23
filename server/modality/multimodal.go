package modality

import "github.com/ollama/ollama/api"

// ChatRequestHasMultimodalPayload reports whether any message carries images, audio, or
// not-yet-expanded video blobs (post-expand: images/video_spans/audio_clips).
//
// Why a dedicated helper: usage breakdown and metrics should not invent modality counts on
// text-only turns; gating here keeps OpenAI prompt_tokens_details sparse when irrelevant.
func ChatRequestHasMultimodalPayload(req *api.ChatRequest) bool {
	if req == nil {
		return false
	}
	for _, msg := range req.Messages {
		if len(msg.Images) > 0 || len(msg.Videos) > 0 || len(msg.VideoSpans) > 0 || len(msg.AudioClips) > 0 || len(msg.PaddedInputIDs) > 0 {
			return true
		}
	}
	return false
}

// ChatRequestHasVideoPayload reports raw video blobs or pre-expanded video_spans in any message.
//
// Why separate from ChatRequestHasMultimodalPayload: still images alone do not require the
// video capability bit or video-specific preflight; SGLang-style pre-expanded spans must
// trigger the same gates as raw videos[] even when ffmpeg will not run.
func ChatRequestHasVideoPayload(req *api.ChatRequest) bool {
	if req == nil {
		return false
	}
	for _, msg := range req.Messages {
		if len(msg.Videos) > 0 || len(msg.VideoSpans) > 0 {
			return true
		}
	}
	return false
}

// lastUserMessageIndex returns the index of the final user message, or -1 if none.
// Used by preflight to scope pre-expanded span/audio estimates to the active turn.
//
// Why latest user only: agents echo multimodal history; counting old expanded frames would
// false-reject follow-up turns. Raw videos[] on any index still expand on this request.
func lastUserMessageIndex(messages []api.Message) int {
	last := -1
	for i, msg := range messages {
		if msg.Role == "user" {
			last = i
		}
	}
	return last
}
