//go:build darwin && uma

package uma_test

import (
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/x/uma"
)

// Lab: grain=op skips coarse HOLD so peer GRAPH can run between Eval one-shots (F0625).
func TestGrainOpAllowsGraphBetweenEvals(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_GRAIN_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_GRAIN_SMOKE=1")
	}
	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("ZEROLLAMA_UMA_GRAIN", "op")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-grain")
	uma.Release()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()
	if !uma.Active() {
		t.Fatal("expected active broker")
	}
	if g := uma.Grain(); g != "op" {
		t.Fatalf("grain want op got %q", g)
	}

	if err := uma.LeaseBegin("decode"); err != nil {
		t.Fatalf("lease begin: %v", err)
	}
	defer uma.LeaseEnd()

	job, err := uma.FormatGraph(1, "chain", "NOP@CPU! ; MARK@GPU?")
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	resp, err := uma.Graph("xuma-grain-gap", job, 15)
	if err != nil {
		t.Fatalf("graph during coarse (should be free under grain=op): %v", err)
	}
	if !strings.Contains(resp, "NOP") {
		t.Fatalf("unexpected: %s", resp)
	}

	ran := 0
	if err := uma.RunGPU(func() { ran++ }); err != nil {
		t.Fatalf("run1: %v", err)
	}
	resp, err = uma.Graph("xuma-grain-mid", job, 15)
	if err != nil {
		t.Fatalf("graph between evals: %v", err)
	}
	if !strings.Contains(resp, "MARK") {
		t.Fatalf("unexpected mid: %s", resp)
	}
	if err := uma.RunGPU(func() { ran++ }); err != nil {
		t.Fatalf("run2: %v", err)
	}
	if ran != 2 {
		t.Fatalf("ran=%d", ran)
	}
}

// Contrast: grain=phase holds GPU; peer GRAPH (separate socket) stays pending.
func TestGrainPhaseQueuesPeerGraph(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_GRAIN_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_GRAIN_SMOKE=1")
	}
	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("ZEROLLAMA_UMA_GRAIN", "phase")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-grain-phase")
	uma.Release()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()
	if uma.Grain() != "phase" {
		t.Fatalf("grain want phase got %q", uma.Grain())
	}

	if err := uma.LeaseBegin("decode"); err != nil {
		t.Fatalf("lease: %v", err)
	}
	defer uma.LeaseEnd()

	// Peer client: SUBMIT GRAPH then JOB — should still be queued while we hold.
	cmd := exec.Command("python3", "-c", `
import socket, time, sys
s = socket.socket(socket.AF_UNIX)
s.settimeout(5)
s.connect("/tmp/uma_daemon.sock")
s.sendall(b"SUBMIT name=peer-graph GRAPH ntok=1 form=chain ; NOP@CPU! ; MARK@GPU?\n")
line = s.recv(512).decode()
if not line.startswith("OK"):
    print("SUBMIT", line); sys.exit(1)
ticket = [p.split("=",1)[1] for p in line.split() if p.startswith("ticket=")][0]
time.sleep(0.4)
s.sendall(f"JOB {ticket}\n".encode())
job = s.recv(512).decode()
s.close()
# Under HOLD: pending/queued, not done with NOP
if "phase=holding" in job or "state=done" in job and "NOP" in job:
    # done with NOP would mean it ran — fail
    if "state=done" in job and "NOP" in job:
        print("RAN_UNDER_HOLD", job); sys.exit(2)
print("QUEUED_OK", job.strip()[:120])
`)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("peer graph probe: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "QUEUED_OK") {
		t.Fatalf("peer output: %s", out)
	}
	time.Sleep(20 * time.Millisecond)
}
