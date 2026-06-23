package modality

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/api"
)

const maxPaddedInputIDsLen = 1 << 20 // 1M tokens — sanity cap for preprocessed client layouts

// PaddedLayoutRunnerStub reports latest-user pretokenized layout after expand.
// Render still uses images + templates until a family processor consumes the ids.
type PaddedLayoutRunnerStub struct {
	Len           int
	HasVideoSpans bool
	SessionKey    string
}

// LatestUserPaddedLayout returns pretokenized layout on the latest user message, if any.
func LatestUserPaddedLayout(req *api.ChatRequest) (PaddedLayoutRunnerStub, bool) {
	if req == nil {
		return PaddedLayoutRunnerStub{}, false
	}
	idx := lastUserMessageIndex(req.Messages)
	if idx < 0 {
		return PaddedLayoutRunnerStub{}, false
	}
	msg := req.Messages[idx]
	if len(msg.PaddedInputIDs) == 0 {
		return PaddedLayoutRunnerStub{}, false
	}
	return PaddedLayoutRunnerStub{
		Len:           len(msg.PaddedInputIDs),
		HasVideoSpans: len(msg.VideoSpans) > 0,
		SessionKey:    ExtractPromptCacheKey(req.Options),
	}, true
}

// LogPaddedLayoutRunnerStub records layout handling for native render (SGLang contract).
func LogPaddedLayoutRunnerStub(model string, stub PaddedLayoutRunnerStub, mode PaddedLayoutConsumeMode) {
	if mode == "" {
		mode = PaddedLayoutConsumeDeferred
	}
	attrs := []any{
		"model", model,
		"padded_input_ids_len", stub.Len,
		"layout_consume", string(mode),
	}
	if stub.HasVideoSpans {
		attrs = append(attrs, "has_video_spans", true)
	}
	if stub.SessionKey != "" {
		attrs = append(attrs, "session_key", stub.SessionKey)
	}
	if mode == PaddedLayoutConsumeDeferred || mode == PaddedLayoutConsumeDeferredHistory {
		attrs = append(attrs, "render_path", "images")
	}
	if mode == PaddedLayoutConsumeQwen3VLHFRunner {
		attrs = append(attrs, "render_path", "prompt_tokens_inject")
	}
	if mode == PaddedLayoutConsumeGemma4ImgRunner {
		attrs = append(attrs, "render_path", "prompt_tokens_inject")
	}
	slog.Info("padded_input_ids runner stub", attrs...)
}

// validatePaddedInputIDs checks SGLang-style pretokenized layouts on pre-expanded messages.
// Why accept before runner wiring: preprocessed clients can fail fast on num_ctx using exact
// layout length; native render path still uses images + templates until mtmd hook exists.
func validatePaddedInputIDs(ids []int) error {
	if len(ids) == 0 {
		return nil
	}
	if len(ids) > maxPaddedInputIDsLen {
		return fmt.Errorf("padded_input_ids length %d exceeds max %d", len(ids), maxPaddedInputIDsLen)
	}
	for i, id := range ids {
		if id < 0 {
			return fmt.Errorf("padded_input_ids[%d] must be non-negative", i)
		}
	}
	return nil
}

// preprocessedLayoutTokenCount returns the token budget to use for preflight/usage when the
// client supplied padded_input_ids (full pretokenized prompt slice for this message).
func preprocessedLayoutTokenCount(msg api.Message) int {
	return len(msg.PaddedInputIDs)
}

// assignPreprocessedUsage sets modality usage fields from padded_input_ids when present.
// Video_spans → video_tokens; otherwise image_tokens. Includes text+vision placeholders in
// the pretokenized layout (SGLang-shaped observability until per-field split exists).
func assignPreprocessedUsage(msg api.Message, out *MultimodalTokenEstimate) {
	n := preprocessedLayoutTokenCount(msg)
	if n <= 0 {
		return
	}
	if len(msg.VideoSpans) > 0 {
		out.VideoTokens = n
		return
	}
	if len(msg.Images) > 0 {
		out.ImageTokens = n
	}
}
