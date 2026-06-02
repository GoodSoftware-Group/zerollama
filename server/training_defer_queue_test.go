package server

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/x/trainingworker"
)

func TestDeferTombstoneSurvivesPromotion(t *testing.T) {
	q := newTrainingDeferQueue(&Server{})
	id, err := q.enqueue("train", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	q.mu.Lock()
	q.markPromotedLocked(id, "job-real-1")
	q.mu.Unlock()

	j, ok := q.lookupSnapshot(id)
	if !ok {
		t.Fatal("defer id should remain after promotion")
	}
	if j.state != deferredStatePromoted || j.promotedID != "job-real-1" {
		t.Fatalf("got state=%q promoted=%q", j.state, j.promotedID)
	}
	promoted, ok := q.promotedJobID(id)
	if !ok || promoted != "job-real-1" {
		t.Fatalf("promotedJobID=%q ok=%v", promoted, ok)
	}
}

func TestDeferMarkFailedKeepsRecord(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_RETRY_MAX", "0")
	q := newTrainingDeferQueue(&Server{})
	id, _ := q.enqueue("train", []byte(`{}`))
	q.mu.Lock()
	q.markFailedOrRetryLocked(id, "python blew up")
	q.mu.Unlock()

	j, ok := q.lookupSnapshot(id)
	if !ok {
		t.Fatal("failed defer job should remain queryable")
	}
	if j.state != deferredStateFailed || j.errMsg != "python blew up" {
		t.Fatalf("got %+v", j)
	}
	raw, err := (&Server{trainingDefer: q}).deferredTrainingJobStatusJSON(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	var wrap struct {
		Job struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		} `json:"job"`
	}
	_ = json.Unmarshal(raw, &wrap)
	if wrap.Job.Status != "failed" || wrap.Job.Error != "python blew up" {
		t.Fatalf("status json: %+v", wrap.Job)
	}
}

func TestDeferCancelWaiting(t *testing.T) {
	q := newTrainingDeferQueue(&Server{})
	id, _ := q.enqueue("train", []byte(`{}`))
	ok, err := q.cancel(id)
	if err != nil || !ok {
		t.Fatalf("cancel: ok=%v err=%v", ok, err)
	}
	j, _ := q.lookupSnapshot(id)
	if j.state != deferredStateCancelled {
		t.Fatalf("state=%q", j.state)
	}
	ok2, err := q.cancel(id)
	if err != nil || !ok2 {
		t.Fatalf("idempotent cancel: ok=%v err=%v", ok2, err)
	}
}

func TestDeferCancelPromotedFails(t *testing.T) {
	q := newTrainingDeferQueue(&Server{})
	id, _ := q.enqueue("train", []byte(`{}`))
	q.mu.Lock()
	q.markPromotedLocked(id, "real")
	q.mu.Unlock()
	_, err := q.cancel(id)
	if err == nil {
		t.Fatal("expected error cancelling promoted defer job")
	}
}

func TestShouldDeferRequiresIdleWait(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "0")
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY", "1")
	err := errors.Join(ErrInferenceBacklogActive)
	if shouldDeferTrainingSubmit(err, TrainingSubmitOptions{Priority: TrainingPriorityNormal}) {
		t.Fatal("defer should require idle-wait env")
	}
}

func TestDeferTombstoneEviction(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_TOMBSTONE_SECS", "1")
	q := newTrainingDeferQueue(&Server{})
	id, _ := q.enqueue("train", []byte(`{}`))
	q.mu.Lock()
	q.markPromotedLocked(id, "job-real")
	q.mu.Unlock()
	if _, ok := q.lookupSnapshot(id); !ok {
		t.Fatal("expected tombstone before eviction")
	}
	time.Sleep(1100 * time.Millisecond)
	q.evictExpired()
	if _, ok := q.lookupSnapshot(id); ok {
		t.Fatal("expected defer tombstone evicted after TTL")
	}
}

func TestDeferPromotionRetry(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_RETRY_MAX", "2")
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_RETRY_SECS", "60")

	q := newTrainingDeferQueue(&Server{})
	id, _ := q.enqueue("train", []byte(`{}`))
	q.mu.Lock()
	q.markFailedOrRetryLocked(id, "transient")
	q.mu.Unlock()

	j, ok := q.lookupSnapshot(id)
	if !ok || j.state != deferredStateWaiting || j.retryCount != 1 {
		t.Fatalf("expected waiting retry, got %+v ok=%v", j, ok)
	}
	if !slices.Contains(q.order, id) {
		t.Fatal("expected job back in order for retry")
	}
}

func TestDeferListWaitingOnlyByDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_LIST_ALL", "0")
	q := newTrainingDeferQueue(&Server{})
	waitID, _ := q.enqueue("train", []byte(`{"a":1}`))
	doneID, _ := q.enqueue("train", []byte(`{"b":2}`))
	q.mu.Lock()
	q.markPromotedLocked(doneID, "real-1")
	q.mu.Unlock()

	entries := q.listEntries()
	if len(entries) != 1 || entries[0].id != waitID {
		t.Fatalf("expected only waiting job in list, got %d", len(entries))
	}
}

func TestMergeDeferredJobsListJSON(t *testing.T) {
	s := &Server{trainingDefer: newTrainingDeferQueue(&Server{})}
	_, _ = s.trainingDefer.enqueue("train", []byte(`{}`))
	merged, err := s.mergeDeferredJobsListJSON([]byte(`{"jobs":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	var root struct {
		Jobs []map[string]any `json:"jobs"`
	}
	_ = json.Unmarshal(merged, &root)
	if len(root.Jobs) != 1 || root.Jobs[0]["defer"] != true {
		t.Fatalf("jobs=%v", root.Jobs)
	}
}

func TestSubmitTrainingJobHighBypassesIdleCheck(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE", "1")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pending.Push(&LlmRequest{ctx: ctx, model: &Model{ModelPath: "/tmp/m"}})

	tw := &trainingworker.Client{}
	s := &Server{sched: sched, training: tw}
	s.trainingDefer = newTrainingDeferQueue(s)

	_, err := s.submitTrainingJob(ctx, "train", []byte(`{}`), TrainingSubmitOptions{
		Priority: TrainingPriorityHigh,
	})
	if err == nil {
		t.Fatal("expected direct submit error without embedded python")
	}
	if errors.Is(err, ErrInferenceBacklogActive) {
		t.Fatal("high priority should not return inference backlog")
	}
}
