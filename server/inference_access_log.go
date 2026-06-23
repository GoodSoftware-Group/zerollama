package server

// Structured inference access logs: request in, phase timings, response out.
// Why phases: separate runner load, template/tokenize, prefill, and decode — agent
// audits showed multi-minute prefill with sub-second prompt_ready; one duration field hid the bottleneck.

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"time"

	"github.com/gin-gonic/gin"
)

const inferenceAccessLogKey = "inference_access_log"

type inferenceQueueSnapshot struct {
	GgmlPending     int
	GgmlActive      int
	GgmlLoaded      int
	GgmlLoadsPaused bool
}

type inferenceAccessMeta struct {
	route              string
	model              string
	stream             bool
	start              time.Time
	queueIn            inferenceQueueSnapshot
	doneReason         string
	promptEvalCount    int
	evalCount          int
	cachedPromptTokens int // L3 / llama-server cache_n; logged on inference response out
	// Prompt sizing — set by routes after chatPrompt/tailTruncate.
	promptTokens    int // tokens actually sent to runner (post-truncation)
	originalTokens  int // tokens before truncation (0 if no truncation occurred)
	messagesDropped int // messages dropped during context window fitting
	// Multimodal heuristic counts (post-expand); mirrors OpenAI prompt_tokens_details.
	imageTokens int
	videoTokens int
	audioTokens int
	// Latest-user pretokenized layout length (SGLang padded_input_ids stub).
	paddedInputIDsLen      int
	paddedLayoutConsume    string
}

func (s *Server) inferenceQueueSnapshot() inferenceQueueSnapshot {
	var snap inferenceQueueSnapshot
	if s == nil || s.sched == nil {
		return snap
	}
	snap.GgmlPending, snap.GgmlActive, snap.GgmlLoaded = s.sched.InferenceBacklog()
	snap.GgmlLoadsPaused = s.sched.loadsPaused.Load()
	return snap
}

func inferenceQueueLogAttrs(snap inferenceQueueSnapshot) []any {
	return []any{
		"ggml_pending", snap.GgmlPending,
		"ggml_active", snap.GgmlActive,
		"ggml_loaded", snap.GgmlLoaded,
		"ggml_loads_paused", snap.GgmlLoadsPaused,
	}
}

func (s *Server) inferenceAccessLogMiddleware(route string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request == nil {
			c.Next()
			return
		}

		model, stream := inferencePeekRequest(c)
		queueIn := s.inferenceQueueSnapshot()
		meta := &inferenceAccessMeta{
			route:   route,
			model:   model,
			stream:  stream,
			start:   time.Now(),
			queueIn: queueIn,
		}
		c.Set(inferenceAccessLogKey, meta)

		attrs := []any{"route", route, "stream", stream}
		if model != "" {
			attrs = append(attrs, "model", model)
		}
		attrs = append(attrs, inferenceQueueLogAttrs(queueIn)...)
		slog.Info("inference request in", attrs...)

		c.Next()
		meta.logResponseOut(c.Writer.Status(), s.inferenceQueueSnapshot())
	}
}

func inferencePeekRequest(c *gin.Context) (model string, stream bool) {
	if c.Request.Body == nil {
		return "", false
	}
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.Request.Body = io.NopCloser(bytes.NewReader(nil))
		return "", false
	}
	c.Request.Body = io.NopCloser(bytes.NewReader(body))
	model, stream = inferencePeekRequestBody(body)
	return model, stream
}

func inferencePeekRequestBody(body []byte) (model string, stream bool) {
	if len(body) == 0 {
		return "", false
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return "", false
	}
	if m, ok := raw["model"]; ok {
		_ = json.Unmarshal(m, &model)
	}
	if s, ok := raw["stream"]; ok {
		_ = json.Unmarshal(s, &stream)
	}
	return model, stream
}

// logInferencePhase emits an INFO-level log for a named phase of inference
// (e.g. "runner_ready", "prompt_ready", "first_token"). Each entry includes
// the model name, phase label, and elapsed time since the request started.
func logInferencePhase(c *gin.Context, phase string, model string, since time.Time) {
	v, _ := c.Get(inferenceAccessLogKey)
	meta, _ := v.(*inferenceAccessMeta)
	var requestElapsed time.Duration
	if meta != nil {
		requestElapsed = time.Since(meta.start)
	}
	slog.Info("inference phase",
		"phase", phase,
		"model", model,
		"phase_elapsed", time.Since(since).Round(time.Millisecond),
		"request_elapsed", requestElapsed.Round(time.Millisecond),
	)
}

// recordInferencePromptSize stores prompt sizing on the access-log meta.
// promptTokens is the post-truncation token count sent to the runner.
// originalTokens is the pre-truncation count (0 when no truncation happened).
// messagesDropped is the number of chat messages dropped to fit the context window.
func recordInferencePromptSize(c *gin.Context, promptTokens, originalTokens, messagesDropped int) {
	v, ok := c.Get(inferenceAccessLogKey)
	if !ok {
		return
	}
	meta, ok := v.(*inferenceAccessMeta)
	if !ok {
		return
	}
	meta.promptTokens = promptTokens
	meta.originalTokens = originalTokens
	meta.messagesDropped = messagesDropped
}

func recordInferenceCompletion(c *gin.Context, doneReason string, promptEvalCount, evalCount, cachedPromptTokens int) {
	v, ok := c.Get(inferenceAccessLogKey)
	if !ok {
		return
	}
	meta, ok := v.(*inferenceAccessMeta)
	if !ok {
		return
	}
	meta.doneReason = doneReason
	meta.promptEvalCount = promptEvalCount
	meta.evalCount = evalCount
	meta.cachedPromptTokens = cachedPromptTokens
}

// recordInferenceMultimodalEstimate stores post-expand modality token heuristics on the
// access-log meta. Why: SGLang/OpenAI expose image_tokens/video_tokens on usage; fleet
// logs should correlate clip expansion with billing-shaped fields without parsing JSON bodies.
func recordInferenceMultimodalEstimate(c *gin.Context, imageTokens, videoTokens, audioTokens int) {
	if imageTokens == 0 && videoTokens == 0 && audioTokens == 0 {
		return
	}
	v, ok := c.Get(inferenceAccessLogKey)
	if !ok {
		return
	}
	meta, ok := v.(*inferenceAccessMeta)
	if !ok {
		return
	}
	meta.imageTokens = imageTokens
	meta.videoTokens = videoTokens
	meta.audioTokens = audioTokens
}

// recordInferencePaddedLayout stores latest-user pretokenized layout on access-log meta.
func recordInferencePaddedLayout(c *gin.Context, paddedInputIDsLen int, layoutConsume string) {
	if paddedInputIDsLen <= 0 {
		return
	}
	v, ok := c.Get(inferenceAccessLogKey)
	if !ok {
		return
	}
	meta, ok := v.(*inferenceAccessMeta)
	if !ok {
		return
	}
	meta.paddedInputIDsLen = paddedInputIDsLen
	if layoutConsume != "" {
		meta.paddedLayoutConsume = layoutConsume
	}
}

func (m *inferenceAccessMeta) logResponseOut(status int, queueOut inferenceQueueSnapshot) {
	if m == nil {
		return
	}
	attrs := []any{
		"route", m.route,
		"stream", m.stream,
		"status", status,
		"duration", time.Since(m.start).Round(time.Millisecond),
	}
	if m.model != "" {
		attrs = append(attrs, "model", m.model)
	}
	attrs = append(attrs, inferenceQueueLogAttrs(queueOut)...)
	if m.queueIn != queueOut {
		attrs = append(attrs,
			"ggml_pending_in", m.queueIn.GgmlPending,
			"ggml_active_in", m.queueIn.GgmlActive,
		)
	}
	if m.doneReason != "" {
		attrs = append(attrs, "done_reason", m.doneReason)
	}
	if m.promptTokens > 0 {
		attrs = append(attrs, "prompt_tokens", m.promptTokens)
	}
	if m.originalTokens > 0 && m.originalTokens != m.promptTokens {
		attrs = append(attrs, "original_tokens", m.originalTokens,
			"truncated_tokens", m.originalTokens-m.promptTokens)
	}
	if m.messagesDropped > 0 {
		attrs = append(attrs, "messages_dropped", m.messagesDropped)
	}
	if m.promptEvalCount > 0 {
		attrs = append(attrs, "prompt_eval_count", m.promptEvalCount)
	}
	if m.evalCount > 0 {
		attrs = append(attrs, "eval_count", m.evalCount)
	}
	if m.cachedPromptTokens > 0 {
		attrs = append(attrs, "cached_prompt_tokens", m.cachedPromptTokens)
	}
	if m.imageTokens > 0 {
		attrs = append(attrs, "image_tokens", m.imageTokens)
	}
	if m.videoTokens > 0 {
		attrs = append(attrs, "video_tokens", m.videoTokens)
	}
	if m.audioTokens > 0 {
		attrs = append(attrs, "audio_tokens", m.audioTokens)
	}
	if m.paddedInputIDsLen > 0 {
		attrs = append(attrs, "padded_input_ids_len", m.paddedInputIDsLen)
		if m.paddedLayoutConsume != "" {
			attrs = append(attrs, "padded_layout_consume", m.paddedLayoutConsume)
		} else {
			attrs = append(attrs, "padded_layout_consume", "deferred")
		}
	}
	slog.Info("inference response out", attrs...)
}
