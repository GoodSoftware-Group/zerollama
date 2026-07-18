package server

// SSE keepalive during long MLX prefill.
// Why: agent HTTP clients (Mercury empty-stream guard) abort when no SSE data:
// frames arrive for ~60s; MLX prefill on 65k+ tokens can exceed that before first token.

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

const defaultStreamKeepaliveInterval = 15 * time.Second

func streamKeepaliveInterval() time.Duration {
	raw := os.Getenv("OLLAMA_STREAM_KEEPALIVE_INTERVAL")
	if raw == "" {
		return defaultStreamKeepaliveInterval
	}
	if raw == "0" || raw == "off" || raw == "false" {
		return 0
	}
	sec, err := strconv.Atoi(raw)
	if err != nil || sec <= 0 {
		if raw != "" {
			slog.Warn("invalid stream keepalive interval; using default",
				"key", "OLLAMA_STREAM_KEEPALIVE_INTERVAL",
				"value", raw,
				"default_sec", int(defaultStreamKeepaliveInterval/time.Second),
			)
		}
		return defaultStreamKeepaliveInterval
	}
	return time.Duration(sec) * time.Second
}

// chatStreamSession streams NDJSON/SSE to the client while inference runs.
// Keepalive status chunks are emitted until StopKeepalive (typically first token).
type chatStreamSession struct {
	ch              chan any
	cancelKeepalive context.CancelFunc
	wg              sync.WaitGroup
	keepaliveWg     sync.WaitGroup
}

func beginChatStream(c *gin.Context, ch chan any, model string) *chatStreamSession {
	sess := &chatStreamSession{ch: ch}
	sess.wg.Add(1)
	go func() {
		defer sess.wg.Done()
		streamResponse(c, ch)
	}()

	if interval := streamKeepaliveInterval(); interval > 0 {
		ctx, cancel := context.WithCancel(c.Request.Context())
		sess.cancelKeepalive = cancel
		sess.keepaliveWg.Add(1)
		go func() {
			defer sess.keepaliveWg.Done()
			runChatStreamKeepalive(ctx, ch, model, interval)
		}()
	}
	return sess
}

func runChatStreamKeepalive(ctx context.Context, ch chan<- any, model string, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if ctx.Err() != nil {
				return
			}
			writeChatStatus(ch, model, "keepalive", "processing", 0, 0)
		}
	}
}

func (s *chatStreamSession) StopKeepalive() {
	if s.cancelKeepalive != nil {
		s.cancelKeepalive()
		s.cancelKeepalive = nil
		s.keepaliveWg.Wait()
	}
}

func (s *chatStreamSession) Wait() {
	s.wg.Wait()
}

func emitSyntheticChatFinish(ch chan<- any, model string) {
	if ch == nil {
		return
	}
	ch <- api.ChatResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		Message:    api.Message{Role: "assistant"},
		Done:       true,
		DoneReason: "stop",
	}
}

func emitSyntheticGenerateFinish(ch chan<- any, model string) {
	if ch == nil {
		return
	}
	ch <- api.GenerateResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC(),
		Done:       true,
		DoneReason: "stop",
	}
}

func enqueueChatStreamError(ch chan<- any, model string, sentDone *bool, errMsg string, status int) {
	enqueueChatStreamErrorExtra(ch, model, sentDone, errMsg, status, inferenceErrorExtra{})
}

func enqueueChatStreamErrorExtra(ch chan<- any, model string, sentDone *bool, errMsg string, status int, extra inferenceErrorExtra) {
	if ch == nil {
		return
	}
	if sentDone != nil && !*sentDone {
		emitSyntheticChatFinish(ch, model)
		*sentDone = true
	}
	if extra.Cause == "" && isHostUnstableError(errMsg) {
		extra.Cause = causeHostUnstable
		metricsIncRunnerCrash()
		metricsIncRequestResult("host_unstable")
	} else if extra.Cause == causeHostUnstable {
		metricsIncRunnerCrash()
		metricsIncRequestResult("host_unstable")
	} else {
		metricsIncRequestResult("error")
	}
	ch <- gin.H(inferenceErrorMap(errMsg, status, extra))
}

func enqueueGenerateStreamError(ch chan<- any, model string, sentDone *bool, errMsg string, status int) {
	enqueueGenerateStreamErrorExtra(ch, model, sentDone, errMsg, status, inferenceErrorExtra{})
}

func enqueueGenerateStreamErrorExtra(ch chan<- any, model string, sentDone *bool, errMsg string, status int, extra inferenceErrorExtra) {
	if ch == nil {
		return
	}
	if sentDone != nil && !*sentDone {
		emitSyntheticGenerateFinish(ch, model)
		*sentDone = true
	}
	if extra.Cause == "" && isHostUnstableError(errMsg) {
		extra.Cause = causeHostUnstable
		metricsIncRunnerCrash()
		metricsIncRequestResult("host_unstable")
	} else if extra.Cause == causeHostUnstable {
		metricsIncRunnerCrash()
		metricsIncRequestResult("host_unstable")
	} else {
		metricsIncRequestResult("error")
	}
	ch <- gin.H(inferenceErrorMap(errMsg, status, extra))
}

func abortChatStream(sess *chatStreamSession, ch chan any, model, errMsg string) {
	abortChatStreamExtra(sess, ch, model, errMsg, inferenceErrorExtra{})
}

func abortChatStreamExtra(sess *chatStreamSession, ch chan any, model, errMsg string, extra inferenceErrorExtra) {
	if sess == nil {
		return
	}
	sess.StopKeepalive()
	if ch != nil && errMsg != "" {
		emitSyntheticChatFinish(ch, model)
		if extra.Cause == "" && isHostUnstableError(errMsg) {
			extra.Cause = causeHostUnstable
			metricsIncRunnerCrash()
			metricsIncRequestResult("host_unstable")
		} else if extra.Cause == "" {
			metricsIncRequestResult("error")
		}
		ch <- gin.H(inferenceErrorMap(errMsg, 0, extra))
	}
	close(ch)
	sess.Wait()
}

// abortStreamingJSON ends a streaming response with an error, or writes JSON when not streaming.
func abortStreamingJSON(c *gin.Context, sess *chatStreamSession, ch chan any, model string, status int, errMsg string) {
	abortStreamingJSONExtra(c, sess, ch, model, status, errMsg, inferenceErrorExtra{})
}

func abortStreamingJSONExtra(c *gin.Context, sess *chatStreamSession, ch chan any, model string, status int, errMsg string, extra inferenceErrorExtra) {
	if sess != nil && ch != nil {
		abortChatStreamExtra(sess, ch, model, errMsg, extra)
		return
	}
	if extra.Cause == "" && isHostUnstableError(errMsg) {
		extra.Cause = causeHostUnstable
	}
	if status == http.StatusServiceUnavailable && extra.RetryAfter == 0 {
		extra.RetryAfter = defaultBusyRetryAfterSec
		c.Header("Retry-After", strconv.Itoa(defaultBusyRetryAfterSec))
	}
	c.JSON(status, inferenceErrorMap(errMsg, 0, extra))
}
