package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/x/trainingworker"
)

func TestSubmitTrainingJobDefersWhenQueueOnBusy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY", "1")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pending.Push(&LlmRequest{
		ctx:   ctx,
		model: &Model{ModelPath: "/tmp/m"},
	})

	s := &Server{sched: sched, training: &trainingworker.Client{}}
	s.trainingDefer = newTrainingDeferQueue(s)
	s.training.SetSubmitHandler(s.handleTrainingSubmitRequest)

	body, _ := json.Marshal(map[string]any{
		"kind":    "train",
		"payload": map[string]any{},
	})
	res, err := s.submitTrainingJob(ctx, "train", body, TrainingSubmitOptions{
		Priority: TrainingPriorityNormal,
	})
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if !res.Queued || res.JobID == "" {
		t.Fatalf("expected queued defer id, got %+v", res)
	}
	if !isDeferredTrainingJobID(res.JobID) {
		t.Fatalf("expected defer- prefix, got %q", res.JobID)
	}
}

func TestSubmitTrainingJobHighPriorityBypassesIdleWait(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pending.Push(&LlmRequest{
		ctx:   ctx,
		model: &Model{ModelPath: "/tmp/m"},
	})

	s := &Server{sched: sched}
	err := s.checkTrainingSubmitAllowed(ctx)
	if err == nil {
		t.Fatal("expected busy without bypass")
	}
	opts := TrainingSubmitOptions{Priority: TrainingPriorityHigh}
	if !opts.Priority.bypassesIdleWait() {
		t.Fatal("high priority should bypass")
	}
}

func TestTrainHTTPSubmitQueuesOnBusy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY", "1")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pending.Push(&LlmRequest{
		ctx:   ctx,
		model: &Model{ModelPath: "/tmp/m"},
	})

	tw := &trainingworker.Client{}
	s := &Server{sched: sched, training: tw}
	s.trainingDefer = newTrainingDeferQueue(s)
	tw.SetSubmitHandler(s.handleTrainingSubmitRequest)

	r := gin.New()
	s.registerTrainingRoutes(r)

	body, _ := json.Marshal(map[string]any{
		"kind":          "train",
		"payload":       map[string]any{},
		"queue_on_busy": true,
	})
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/train/jobs", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["queued"] != true {
		t.Fatalf("expected queued=true, got %v", out)
	}
}

func TestDeferredTrainingJobStatusJSON(t *testing.T) {
	s := &Server{trainingDefer: newTrainingDeferQueue(&Server{})}
	id, err := s.trainingDefer.enqueue("train", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := s.deferredTrainingJobStatusJSON(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	var wrap struct {
		Job struct {
			JobID  string `json:"jobId"`
			Status string `json:"status"`
			Defer  bool   `json:"defer"`
		} `json:"job"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		t.Fatal(err)
	}
	if wrap.Job.JobID != id || wrap.Job.Status != "pending" || !wrap.Job.Defer {
		t.Fatalf("unexpected job: %+v", wrap.Job)
	}
}

func TestSubmitTrainingJobRejectsOutsideWindow(t *testing.T) {
	now := time.Now().UTC()
	h := (now.Hour() + 2) % 24
	end := (h + 1) % 24
	t.Setenv("ZEROLLAMA_TRAINING_ALLOWED_WINDOW", fmt.Sprintf("%02d:00-%02d:00", h, end))
	t.Setenv("ZEROLLAMA_TRAINING_WINDOW_TZ", "UTC")

	s := &Server{training: &trainingworker.Client{}}
	_, err := s.submitTrainingJob(context.Background(), "train", []byte(`{}`), TrainingSubmitOptions{})
	if !errors.Is(err, ErrTrainingOutsideWindow) {
		t.Fatalf("expected outside window, got %v", err)
	}
}

func TestSubmitTrainingJobRejectsMisconfiguredWindow(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_ALLOWED_WINDOW", "bad-window")
	s := &Server{training: &trainingworker.Client{}}
	_, err := s.submitTrainingJob(context.Background(), "train", []byte(`{}`), TrainingSubmitOptions{})
	if !errors.Is(err, ErrTrainingWindowMisconfigured) {
		t.Fatalf("expected misconfigured window, got %v", err)
	}
}

func TestShouldDeferTrainingSubmitOutsideWindow(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_ALLOWED_WINDOW", "22:00-06:00")
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY", "1")
	err := fmt.Errorf("wrap: %w", ErrTrainingOutsideWindow)
	if !shouldDeferTrainingSubmit(err, TrainingSubmitOptions{Priority: TrainingPriorityNormal}) {
		t.Fatal("expected defer when queue on busy")
	}
}

func TestTrainingDeferQueueDrainBlockedWhileBusy(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pending.Push(&LlmRequest{
		ctx:   ctx,
		model: &Model{ModelPath: "/tmp/m"},
	})

	s := &Server{sched: sched, training: &trainingworker.Client{}}
	s.trainingDefer = newTrainingDeferQueue(s)
	id, _ := s.trainingDefer.enqueue("train", []byte(`{}`))
	s.trainingDefer.drainOnce(ctx)
	if j, ok := s.trainingDefer.lookupSnapshot(id); !ok || j.state != deferredStateWaiting {
		t.Fatal("defer job should remain waiting while inference busy")
	}
}
