//go:build darwin && uma

package uma_test

import (
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/x/uma"
)

// Lab smoke: HOLD_GPU ∥ HOLD_ANE via x/uma LeaseBeginUnit (broker F0390).
// Requires uma_daemon with HOLD_ANE. Skip if broker down or UMA off.
func TestMultiUnitLeaseGPUAndANE(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_MULTIUNIT_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_MULTIUNIT_SMOKE=1")
	}
	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-multiunit")
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()
	if !uma.Active() {
		t.Fatal("expected active broker gate")
	}

	if err := uma.LeaseBeginUnit("gpu", "smoke"); err != nil {
		t.Fatalf("gpu lease: %v", err)
	}
	if err := uma.LeaseBeginUnit("ane", "smoke"); err != nil {
		uma.LeaseEndUnit("gpu")
		t.Fatalf("ane lease (upgrade uma_daemon for HOLD_ANE?): %v", err)
	}

	// Both held — RunUnit nested should not re-SUBMIT.
	ran := false
	if err := uma.RunUnit("gpu", func() { ran = true }); err != nil {
		t.Fatalf("run gpu: %v", err)
	}
	if !ran {
		t.Fatal("gpu job did not run")
	}
	ran = false
	if err := uma.RunUnit("ane", func() { ran = true }); err != nil {
		t.Fatalf("run ane: %v", err)
	}
	if !ran {
		t.Fatal("ane job did not run")
	}

	time.Sleep(50 * time.Millisecond)
	uma.LeaseEndUnit("ane")
	uma.LeaseEndUnit("gpu")
}
