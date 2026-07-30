package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestApplyRequestTimeout(t *testing.T) {
	t.Parallel()
	parent := context.Background()

	ctx, cancel := applyRequestTimeout(parent, nil)
	if cancel != nil {
		t.Fatal("nil timeout must not cancel")
	}
	if ctx != parent {
		t.Fatal("nil timeout must return parent")
	}

	zero := &api.Duration{Duration: 0}
	ctx, cancel = applyRequestTimeout(parent, zero)
	if cancel != nil {
		t.Fatal("zero timeout must not cancel")
	}

	d := &api.Duration{Duration: 50 * time.Millisecond}
	ctx, cancel = applyRequestTimeout(parent, d)
	if cancel == nil {
		t.Fatal("positive timeout must return cancel")
	}
	defer cancel()
	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("expected deadline")
	}
	if time.Until(deadline) > 100*time.Millisecond {
		t.Fatalf("deadline too far: %v", deadline)
	}
	select {
	case <-ctx.Done():
		if !errors.Is(ctx.Err(), context.DeadlineExceeded) {
			t.Fatalf("err=%v", ctx.Err())
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timeout did not fire")
	}
}

func TestWriteRequestTimeout(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	to := &api.Duration{Duration: 2 * time.Second}
	writeRequestTimeout(c, to)
	if w.Code != http.StatusGatewayTimeout {
		t.Fatalf("status=%d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "request timeout") || !strings.Contains(body, "timeout_seconds") {
		t.Fatalf("body=%s", body)
	}
}

func TestIsRequestTimeout(t *testing.T) {
	t.Parallel()
	if isRequestTimeout(nil) {
		t.Fatal("nil")
	}
	if isRequestTimeout(context.Canceled) {
		t.Fatal("canceled is not timeout")
	}
	if !isRequestTimeout(context.DeadlineExceeded) {
		t.Fatal("want DeadlineExceeded")
	}
}

func TestHandleScheduleError_DeadlineExceeded(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Set("request_timeout", &api.Duration{Duration: 3 * time.Second})
	handleScheduleError(c, "m", context.DeadlineExceeded)
	if w.Code != http.StatusGatewayTimeout {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "timeout_seconds") {
		t.Fatalf("body=%s", w.Body.String())
	}
}
