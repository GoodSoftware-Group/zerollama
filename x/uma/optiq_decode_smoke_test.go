//go:build darwin && uma

package uma_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/uma"
)

// TestOptiqGraphDecodeAlwaysOn — F0685 session-resident every-gap GRAPH + owned InProjZ GEMV.
func TestOptiqGraphDecodeAlwaysOn(t *testing.T) {
	if os.Getenv("ZEROLLAMA_UMA_OPTIQ_DECODE_SMOKE") != "1" {
		t.Skip("set ZEROLLAMA_UMA_OPTIQ_DECODE_SMOKE=1")
	}
	_ = os.Setenv("ZEROLLAMA_UMA_SCHED", "require")
	_ = os.Setenv("ZEROLLAMA_UMA_OPTIQ_GRAPH_DECODE", "owned")
	_ = os.Setenv("UMA_JOB_NAME", "xuma-optiq-decode")
	uma.Release()
	uma.ResetOptiqDecodeForTest()
	if err := uma.Acquire(); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer uma.Release()
	defer uma.ResetOptiqDecodeForTest()

	if err := uma.EnsureOptiqDecodeSession(); err != nil {
		t.Fatalf("session: %v", err)
	}

	dump := os.Getenv("UMA_OPTIQ_DUMP_DIR")
	if dump == "" {
		dump = "/tmp/uma_optiq_live_dump"
	}
	xb, err := os.ReadFile(filepath.Join(dump, "x.bin"))
	if err != nil {
		t.Fatalf("x.bin: %v", err)
	}
	x := make([]float32, len(xb)/4)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(xb[i*4:]))
	}

	for i := 0; i < 3; i++ {
		if err := uma.MaybeOptiqGraphDecodeStep(); err != nil {
			t.Fatalf("gap step %d: %v", i, err)
		}
		z, err := uma.OwnedInProjZGemv(x)
		if err != nil {
			t.Fatalf("owned gemv %d: %v", i, err)
		}
		if len(z) != 4096 {
			t.Fatalf("owned z len=%d", len(z))
		}
	}
	if uma.OptiqDecodeSteps() != 3 {
		t.Fatalf("gap steps=%d want 3", uma.OptiqDecodeSteps())
	}
	if uma.OptiqOwnedConsumed() < 3 {
		t.Fatalf("owned consumed=%d want >=3", uma.OptiqOwnedConsumed())
	}
	t.Log("PASS: always-on gap GRAPH ×3 + owned InProjZ GEMV ×3")
}
