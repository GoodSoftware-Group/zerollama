package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestAbortIfTrainingBusyNoTrainingClient(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING", "1")

	s := &Server{}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/generate", nil)
	if s.abortIfTrainingBusy(c) {
		t.Fatal("expected no abort without training client")
	}
}

func TestUpdateSchedInferencePauses(t *testing.T) {
	t.Setenv("ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING", "0")
	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "1")
	t.Setenv("ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG", "2")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"waiting":2,"running":0}`))
	}))
	defer srv.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)

	schedCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(schedCtx)
	s := &Server{sched: sched}

	s.updateSchedInferencePauses(context.Background())
	deadline := time.Now().Add(2 * time.Second)
	for !sched.loadsPaused.Load() {
		if time.Now().After(deadline) {
			t.Fatal("scheduler did not pause on runtime backlog")
		}
		time.Sleep(10 * time.Millisecond)
	}

	t.Setenv("ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY", "0")
	s.updateSchedInferencePauses(context.Background())
	for sched.loadsPaused.Load() {
		if time.Now().After(deadline) {
			t.Fatal("scheduler did not resume")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
