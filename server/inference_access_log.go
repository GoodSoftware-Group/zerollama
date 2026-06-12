package server

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
	GgmlPending      int
	GgmlActive       int
	GgmlLoaded       int
	GgmlLoadsPaused  bool
}

type inferenceAccessMeta struct {
	route           string
	model           string
	stream          bool
	start           time.Time
	queueIn         inferenceQueueSnapshot
	doneReason      string
	promptEvalCount int
	evalCount       int
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

func recordInferenceCompletion(c *gin.Context, doneReason string, promptEvalCount, evalCount int) {
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
	if m.promptEvalCount > 0 {
		attrs = append(attrs, "prompt_eval_count", m.promptEvalCount)
	}
	if m.evalCount > 0 {
		attrs = append(attrs, "eval_count", m.evalCount)
	}
	slog.Info("inference response out", attrs...)
}
