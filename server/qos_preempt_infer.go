package server

import (
	"context"
	"errors"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/modality"
)

func isContextCanceled(err error) bool {
	return err != nil && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded))
}

// bindInferPreemptCancel registers a cancel func on the MLX/session gate so
// interactive soft-preempt can abort this generate mid-stream (M15f).
func (s *Server) bindInferPreemptCancel(ctx context.Context, m *Model, opts map[string]any) (context.Context, context.CancelFunc) {
	if s == nil || s.sched == nil || m == nil || ctx == nil {
		return ctx, nil
	}
	key := modality.ExtractPromptCacheKey(opts)
	if key == "" {
		return ctx, nil
	}
	inferCtx, cancel := context.WithCancel(ctx)
	s.sched.mlxGate.bindPreemptCancel(schedulerModelKey(m), key, cancel)
	return inferCtx, cancel
}

// maybeEnqueueGeneratePreempted emits a done chunk with done_reason=preempted when
// soft-preempt canceled the infer context. Returns true when it handled the error.
func (s *Server) maybeEnqueueGeneratePreempted(
	ch chan<- any,
	m *Model,
	opts map[string]any,
	modelName string,
	partial string,
	sentDone *bool,
	checkpointStart, checkpointLoaded time.Time,
	ggmlCtx *api.GgmlNumCtx,
) bool {
	if s == nil || s.sched == nil || m == nil || ch == nil {
		return false
	}
	reason := s.sched.mlxGate.takePreemptReason(schedulerModelKey(m), modality.ExtractPromptCacheKey(opts))
	if reason == "" {
		return false
	}
	res := api.GenerateResponse{
		Model:           modelName,
		CreatedAt:       time.Now().UTC(),
		Response:        partial,
		Done:            true,
		DoneReason:      "preempted",
		PreemptedReason: reason,
		Metrics: api.Metrics{
			TotalDuration: time.Since(checkpointStart),
		},
	}
	if !checkpointLoaded.IsZero() {
		res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
	}
	applyGgmlNumCtxResponse(&res, ggmlCtx)
	if sentDone != nil {
		*sentDone = true
	}
	ch <- res
	return true
}

// maybeEnqueueChatPreempted is the chat counterpart of maybeEnqueueGeneratePreempted.
func (s *Server) maybeEnqueueChatPreempted(
	ch chan<- any,
	m *Model,
	opts map[string]any,
	modelName string,
	_ string, // reserved content accumulator
	thinkingPartial string,
	sentDone *bool,
	checkpointStart, checkpointLoaded time.Time,
	ggmlCtx *api.GgmlNumCtx,
) bool {
	if s == nil || s.sched == nil || m == nil || ch == nil {
		return false
	}
	reason := s.sched.mlxGate.takePreemptReason(schedulerModelKey(m), modality.ExtractPromptCacheKey(opts))
	if reason == "" {
		return false
	}
	res := api.ChatResponse{
		Model:     modelName,
		CreatedAt: time.Now().UTC(),
		Message: api.Message{
			Role:     "assistant",
			Thinking: thinkingPartial,
		},
		Done:            true,
		DoneReason:      "preempted",
		PreemptedReason: reason,
		Metrics: api.Metrics{
			TotalDuration: time.Since(checkpointStart),
		},
	}
	if !checkpointLoaded.IsZero() {
		res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
	}
	applyGgmlNumCtxChatResponse(&res, ggmlCtx)
	if sentDone != nil {
		*sentDone = true
	}
	ch <- res
	return true
}
