package discover

import "testing"

func TestDraftDaemonBenchIters(t *testing.T) {
	if draftDaemonBenchIters(true) != 10 {
		t.Fatal("quick iters")
	}
	if draftDaemonBenchIters(false) != 30 {
		t.Fatal("default iters")
	}
}

func TestKernelReuseCheck(t *testing.T) {
	ready := ANEDraftDaemonReady{CompileCount: 1}
	first := ANEDraftDaemonBench{CompileCount: 1, EvalCount: 10}
	second := ANEDraftDaemonBench{CompileCount: 1, EvalCount: 20}
	kernelReused := ready.CompileCount == 1 &&
		first.CompileCount == 1 &&
		second.CompileCount == 1 &&
		second.EvalCount == first.EvalCount+10
	if !kernelReused {
		t.Fatal("expected kernel reuse")
	}
}
