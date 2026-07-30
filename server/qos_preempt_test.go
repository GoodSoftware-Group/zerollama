package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestWrapQoSDeferAbort(t *testing.T) {
	t.Parallel()
	if wrapQoSDeferAbort(nil, "x") != nil {
		t.Fatal("nil err")
	}
	err := wrapQoSDeferAbort(context.Canceled, "")
	if !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if preemptedReasonFromErr(err) != "" {
		t.Fatal("empty policy")
	}
	err = wrapQoSDeferAbort(context.Canceled, "lower_wait_interactive")
	if !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if got := preemptedReasonFromErr(err); got != "lower_wait_interactive" {
		t.Fatalf("got %q", got)
	}
}

func TestHandleScheduleError_PreemptedReason(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	err := wrapQoSDeferAbort(context.Canceled, "lower_wait_interactive")
	handleScheduleError(c, "m", err)
	if w.Code != 499 {
		t.Fatalf("status=%d", w.Code)
	}
	if !strings.Contains(w.Body.String(), `"preempted_reason":"lower_wait_interactive"`) {
		t.Fatalf("body=%s", w.Body.String())
	}
	_ = http.StatusOK
}
