package server

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestClassifyTransientMLXError(t *testing.T) {
	hold := errors.New("mlx runner failed: mlx: HOLD_GPU failed\nError: uma lease begin (gpu): HOLD_GPU failed (exit: exit status 1)")
	got := classifyTransientMLXError(hold)
	if !errors.Is(got, ErrUmaGPULease) {
		t.Fatalf("want ErrUmaGPULease, got %v", got)
	}
	killed := errors.New("mlx runner exited unexpectedly: signal: killed")
	got = classifyTransientMLXError(killed)
	if !errors.Is(got, ErrMLXRunnerJetsam) {
		t.Fatalf("want ErrMLXRunnerJetsam, got %v", got)
	}
	other := errors.New("weights corrupt")
	if classifyTransientMLXError(other) != other {
		t.Fatal("non-transient must pass through")
	}
}

func TestLoadCooldownSkipsUmaLease(t *testing.T) {
	t.Setenv("ZEROLLAMA_LOAD_COOLDOWN", "10s")
	s := InitScheduler(t.Context())
	err := fmt.Errorf("%w: raw", ErrUmaGPULease)
	s.noteLoadFailure("k", err)
	if s.cooldownErr("k") != nil {
		t.Fatal("HOLD_GPU / uma lease must not enter load cooldown")
	}
	s.noteLoadFailure("k2", fmt.Errorf("%w: raw", ErrMLXRunnerJetsam))
	if s.cooldownErr("k2") != nil {
		t.Fatal("jetsam must not enter load cooldown")
	}
}

func TestHandleScheduleErrorUmaLease(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	handleScheduleError(c, "gemma4:26b-optiq", fmt.Errorf("%w: detail", ErrUmaGPULease))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	body := w.Body.String()
	if !strings.Contains(body, "gpu_lease") {
		t.Fatalf("want error_code=gpu_lease in %s", body)
	}
	if strings.Contains(strings.ToLower(body), "not found") {
		t.Fatalf("must not look like model-not-found: %s", body)
	}
}

func TestHandleScheduleErrorJetsam(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	handleScheduleError(c, "gemma4:26b-optiq", errors.New("mlx runner exited unexpectedly: signal: killed"))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "mlx_jetsam") {
		t.Fatalf("want error_code=mlx_jetsam in %s", w.Body.String())
	}
}
