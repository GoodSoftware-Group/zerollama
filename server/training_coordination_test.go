package server

import (
	"context"
	"testing"
)

func TestTrainingDeferCoordinationStats(t *testing.T) {
	q := newTrainingDeferQueue(&Server{})
	_, err := q.enqueue("train", []byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	st := q.coordinationStats()
	if st["defer_waiting"].(int) != 1 {
		t.Fatalf("waiting %v", st["defer_waiting"])
	}
	if st["defer_tracked"].(int) != 1 {
		t.Fatalf("tracked %v", st["defer_tracked"])
	}
}

func TestPushRuntimeCoordinationNilSafe(t *testing.T) {
	var s *Server
	s.pushRuntimeCoordination(context.Background())
}
