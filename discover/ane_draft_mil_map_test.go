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
	slots, matched, required := matchMILSlots(eagle3MILSlotSpec, tensors, true)
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

func TestMatchQwen35DrafterMILSlots(t *testing.T) {
	tensors := []*ggml.Tensor{
		{Name: "blk.0.ffn_gate.weight", Shape: []uint64{768, 2688}},
		{Name: "blk.0.ffn_norm.weight", Shape: []uint64{768}},
	}
	slots, matched, required := matchMILSlots(qwen35DrafterMILSlotSpec, tensors, true)
	if required != 2 {
		t.Fatalf("phase3 required = %d", required)
	}
	if matched != 3 {
		t.Fatalf("matched = %d want 3 (2 tensors + proxy)", matched)
	}
	if len(slots) != 3 {
		t.Fatalf("slots = %d", len(slots))
	}
}

func TestMatchDflashMILSlots(t *testing.T) {
	tensors := []*ggml.Tensor{
		{Name: "dflash_fc.weight", Shape: []uint64{25600, 5120}},
		{Name: "dflash_hidden_norm.weight", Shape: []uint64{5120}},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{5120, 5120}},
		{Name: "blk.0.ffn_gate.weight", Shape: []uint64{5120, 27648}},
	}
	slots, matched, required := matchMILSlots(dflashMILSlotSpec, tensors, true)
	if required != 4 {
		t.Fatalf("phase3 required = %d", required)
	}
	if matched != 5 {
		t.Fatalf("matched = %d want 5 (4 tensors + proxy)", matched)
	}
	if len(slots) != 5 {
		t.Fatalf("slots = %d", len(slots))
	}
}

func TestDefaultProxyConvTensorForArch(t *testing.T) {
	if got := DefaultProxyConvTensorForArch("dflash-draft"); got != "blk.0.ffn_gate.weight" {
		t.Fatalf("dflash proxy tensor = %q", got)
	}
	if got := DefaultProxyConvTensorForArch("eagle3"); got != "blk.0.ffn_gate.weight" {
		t.Fatalf("eagle3 proxy tensor = %q", got)
	}
}

func TestDraftMILWeightBlobBytes(t *testing.T) {
	if draftMILWeightBlobBytes(256) != 64+64+256*256*2 {
		t.Fatal("blob size mismatch")
	}
}
