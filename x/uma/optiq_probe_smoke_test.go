//go:build darwin && uma

package uma_test

import (
	"os"
	"testing"

	"github.com/ollama/ollama/x/uma"
)

// TestOptiqMaybeProbe — F0635 helper used by mlxrunner decode gap.
func TestOptiqMaybeProbe(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_OPTIQ_PROBE_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_OPTIQ_PROBE_SMOKE=1")
	}
	// Empty UMA_OPTIQ_DUMP_DIR → probe defaults to /tmp/uma_optiq_live_dump (F0636).
	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_PROBE", "require")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-optiq-probe")
	uma.Release()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()

	did, err := uma.MaybeProbeOptiqLiveChain()
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	if !did {
		t.Fatal("expected probe to run")
	}
	// second call is a no-op (once)
	did2, err2 := uma.MaybeProbeOptiqLiveChain()
	if err2 != nil {
		t.Fatalf("probe2: %v", err2)
	}
	if did2 {
		t.Fatal("expected second probe skipped")
	}
	t.Log("PASS: MaybeProbeOptiqLiveChain once")
}
