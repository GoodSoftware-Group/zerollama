package server

import (
	"context"
	"testing"
)

func TestCoordinationWorkloadFields(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	s := &Server{sched: sched}
	sched.pending.Push(&LlmRequest{ctx: ctx, model: &Model{ModelPath: "/tmp/m"}})
	fields := s.coordinationWorkloadFields(ctx, runtimeHealthSnapshot{})
	if fields["sched_pending"].(int) != 1 {
		t.Fatalf("sched_pending=%v", fields["sched_pending"])
	}
	if fields["sched_active"].(int) != 0 {
		t.Fatalf("sched_active=%v", fields["sched_active"])
	}
}
