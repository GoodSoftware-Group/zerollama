package server

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// applyRequestTimeout wraps parent with a deadline when timeout is set and > 0.
// WHY: Hermes needs server-enforced per-call bounds; client disconnect alone
// leaves stuck generations holding QoS/slots until the HTTP client gives up.
// Returns cancel (may be nil) — caller must defer cancel when non-nil.
func applyRequestTimeout(parent context.Context, timeout *api.Duration) (context.Context, context.CancelFunc) {
	if timeout == nil || timeout.Duration <= 0 {
		return parent, nil
	}
	return context.WithTimeout(parent, timeout.Duration)
}

// writeRequestTimeout responds when the per-call deadline fires (distinct from 499 cancel).
// WHY 504: DeadlineExceeded is server policy; Canceled is client abandon — Hermes
// retry logic treats them differently (escalate vs drop).
func writeRequestTimeout(c *gin.Context, timeout *api.Duration) {
	body := gin.H{"error": "request timeout"}
	if timeout != nil && timeout.Duration > 0 {
		body["timeout_seconds"] = timeout.Duration.Seconds()
	}
	c.JSON(http.StatusGatewayTimeout, body)
}

func isRequestTimeout(err error) bool {
	return err != nil && errors.Is(err, context.DeadlineExceeded)
}

func timeoutDurationOrZero(timeout *api.Duration) time.Duration {
	if timeout == nil {
		return 0
	}
	return timeout.Duration
}
