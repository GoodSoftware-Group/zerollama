package server

import (
	"context"
	"testing"
)

func TestTrainingDeferPausesGgml(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "1")
	s := &Server{trainingDefer: newTrainingDeferQueue(&Server{})}
	if s.trainingDeferPausesGgml() {
		t.Fatal("empty defer queue should not pause")
	}
	_, err := s.trainingDefer.enqueue("train", []byte(`{}`), TrainingPriorityNormal)
	if err != nil {
		t.Fatal(err)
	}
	if !s.trainingDeferPausesGgml() {
		t.Fatal("waiting defer should pause ggml loads")
	}
	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "0")
	if s.trainingDeferPausesGgml() {
		t.Fatal("explicit off must disable")
	}
}

func TestInferencePausesGgmlIncludesDefer(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "1")
	t.Setenv("ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING", "0")
	s := &Server{trainingDefer: newTrainingDeferQueue(&Server{})}
	_, _ = s.trainingDefer.enqueue("train", []byte(`{}`), TrainingPriorityLow)
	h := runtimeHealthSnapshot{}
	if !s.inferencePausesGgml(context.Background(), h) {
		t.Fatal("defer waiting should pause via inferencePausesGgml")
	}
}

func TestDeferEnqueuePausesGgmlImmediately(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "1")
	t.Setenv("ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING", "0")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "") // no runtime probe in unit test
	s := &Server{sched: &Scheduler{pending: newPendingQueue(8)}}
	s.trainingDefer = newTrainingDeferQueue(s)
	if s.sched.LoadsPaused() {
		t.Fatal("sched should start unpaused")
	}
	// Same path as tryDeferTrainingSubmit after enqueue.
	id, err := s.trainingDefer.enqueue("train", []byte(`{}`), TrainingPriorityNormal)
	if err != nil {
		t.Fatal(err)
	}
	s.syncRuntimeCoordinationAndPauses(context.Background())
	if !s.sched.LoadsPaused() {
		t.Fatal("expected PauseNewLoads immediately after defer enqueue + sync")
	}
	// Cancel last waiter → resume (cancelDeferredTrainingJob syncs again).
	ok, err := s.cancelDeferredTrainingJob(id)
	if err != nil || !ok {
		t.Fatalf("cancel: ok=%v err=%v", ok, err)
	}
	if s.sched.LoadsPaused() {
		t.Fatal("expected ResumeLoads after last defer cancelled")
	}
}
