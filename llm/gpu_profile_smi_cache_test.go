package llm

import (
	"testing"
	"time"
)

// TestNvidiaSMIIdentityCached pins the per-request probe fix: a hit runs nvidia-smi once
// per process, and a miss is retried only after nvidiaSMIRetry.
func TestNvidiaSMIIdentityCached(t *testing.T) {
	reset := func() {
		c := &nvidiaSMIIdentity
		c.Lock()
		c.name, c.vramGB, c.ok, c.at = "", 0, false, time.Time{}
		c.Unlock()
	}
	orig := nvidiaSMIGPUIdentityUncached
	t.Cleanup(func() { nvidiaSMIGPUIdentityUncached = orig; reset() })

	calls := 0
	result := func() (string, float64) { return "", 0 }
	nvidiaSMIGPUIdentityUncached = func() (string, float64) { calls++; return result() }

	reset()
	for range 3 {
		nvidiaSMIGPUIdentity()
	}
	if calls != 1 {
		t.Fatalf("miss: %d probes in the retry window, want 1", calls)
	}

	nvidiaSMIIdentity.at = time.Now().Add(-2 * nvidiaSMIRetry) // window expired
	result = func() (string, float64) { return "NVIDIA GeForce RTX 4090", 24 }
	if name, vram := nvidiaSMIGPUIdentity(); name != "NVIDIA GeForce RTX 4090" || vram != 24 || calls != 2 {
		t.Fatalf("retry: got %q %v after %d probes", name, vram, calls)
	}

	nvidiaSMIIdentity.at = time.Now().Add(-2 * nvidiaSMIRetry)
	for range 3 {
		nvidiaSMIGPUIdentity()
	}
	if calls != 2 {
		t.Fatalf("hit: %d probes, want the hit kept for the process", calls)
	}
}
