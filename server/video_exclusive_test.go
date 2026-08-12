package server

import (
	"context"
	"testing"
	"time"
)

func TestVideoExclusiveRequested(t *testing.T) {
	if !videoExclusiveRequested(nil) {
		t.Fatal("default on")
	}
	if !videoExclusiveRequested(map[string]any{}) {
		t.Fatal("empty opts on")
	}
	if videoExclusiveRequested(map[string]any{
		"zerollama": map[string]any{"fulfillment": "none"},
	}) {
		t.Fatal("none opts out")
	}
	if videoExclusiveRequested(map[string]any{
		"zerollama": map[string]any{"fulfillment": "off"},
	}) {
		t.Fatal("off opts out")
	}
	if !videoExclusiveRequested(map[string]any{
		"zerollama": map[string]any{"fulfillment": "exclusive"},
	}) {
		t.Fatal("exclusive stays on")
	}
}

func TestVideoJobStatusTerminal(t *testing.T) {
	if !videoJobStatusTerminal([]byte(`{"job":{"status":"completed"}}`)) {
		t.Fatal("completed")
	}
	if !videoJobStatusTerminal([]byte(`{"job":{"status":"failed"}}`)) {
		t.Fatal("failed")
	}
	if videoJobStatusTerminal([]byte(`{"job":{"status":"in_progress"}}`)) {
		t.Fatal("in_progress not terminal")
	}
}

func TestAcquireReleaseVideoExclusive(t *testing.T) {
	s := &Server{sched: &Scheduler{
		mlxGate: *newMLXAgentGate(),
		pending: newPendingQueue(8),
	}}

	s.acquireVideoExclusiveGPU(context.Background(), "job-a")
	if !s.videoExclusiveActive() {
		t.Fatal("expected active after acquire")
	}
	hold, ok := s.sched.mlxGate.fulfillmentActive(time.Now())
	if !ok || !hold.mode.Exclusive() {
		t.Fatalf("expected exclusive fulfillment hold, got ok=%v hold=%+v", ok, hold)
	}
	if !s.trainingOccupiesGPU(context.Background()) {
		t.Fatal("trainingOccupiesGPU should be true while video exclusive")
	}

	// Duplicate acquire is a no-op.
	s.acquireVideoExclusiveGPU(context.Background(), "job-a")

	s.releaseVideoExclusiveGPU("job-a")
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && s.videoExclusiveActive() {
		time.Sleep(20 * time.Millisecond)
	}
	if s.videoExclusiveActive() {
		t.Fatal("expected inactive after release")
	}
	if _, ok := s.sched.mlxGate.fulfillmentActive(time.Now()); ok {
		t.Fatal("fulfillment should be cleared")
	}
}

func TestAcquireSkipsDeferIDs(t *testing.T) {
	s := &Server{sched: &Scheduler{
		mlxGate: *newMLXAgentGate(),
		pending: newPendingQueue(8),
	}}
	s.acquireVideoExclusiveGPU(context.Background(), "defer-abc")
	if s.videoExclusiveActive() {
		t.Fatal("defer ids must not latch exclusive")
	}
}
