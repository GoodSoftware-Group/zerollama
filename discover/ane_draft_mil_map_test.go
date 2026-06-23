package discover

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestANEDraftSidecarCandidatesEliza2B(t *testing.T) {
	paths := ANEDraftSidecarCandidates("eliza-1-2b-dflash")
	if len(paths) < 3 {
		t.Fatalf("expected cache candidate, got %v", paths)
	}
	found := false
	for _, p := range paths {
		if strings.Contains(p, "drafter-2b.gguf") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("missing drafter-2b path in %v", paths)
	}
}

func TestMatchEagle3MILSlots(t *testing.T) {
	tensors := []*ggml.Tensor{
		{Name: "fc.weight", Shape: []uint64{6144, 256}},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{512, 256}},
		{Name: "blk.0.ffn_gate.weight", Shape: []uint64{256, 1024}},
	}
	slots, matched, required := matchEagle3MILSlots(tensors, true)
	if required != 3 {
		t.Fatalf("phase3 required = %d", required)
	}
	if matched != 4 {
		t.Fatalf("matched = %d want 4 (3 tensors + proxy)", matched)
	}
	if len(slots) != 4 {
		t.Fatalf("slots = %d", len(slots))
	}
}

func TestDraftMILWeightBlobBytes(t *testing.T) {
	if draftMILWeightBlobBytes(256) != 64+64+256*256*2 {
		t.Fatal("blob size mismatch")
	}
}
