package cmd

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/discover"
)

func TestNewFlashMoEResolveCommand(t *testing.T) {
	c := NewFlashMoEResolveCommand()
	if c == nil || c.Use != "flash-moe-resolve" {
		t.Fatalf("command = %+v", c)
	}
	if c.Flags().Lookup("print-env") == nil {
		t.Fatal("missing --print-env")
	}
}

func TestFlashMoEResolveRowSlotBank(t *testing.T) {
	tiny := flashMoEResolveRowFrom(discover.FlashMoEInventoryEntry{ExpertCount: 0}, 128)
	if tiny.FreetokenSlotBank != 1 || tiny.RecommendSlotBank != 1 {
		t.Fatalf("experts=0 %+v", tiny)
	}
	big := flashMoEResolveRowFrom(discover.FlashMoEInventoryEntry{ExpertCount: 256}, 128)
	if big.FreetokenSlotBank < 1 || big.FreetokenSlotBank > 256 {
		t.Fatalf("experts=256 bank=%d", big.FreetokenSlotBank)
	}
	if big.RecommendSlotBank != big.FreetokenSlotBank {
		t.Fatalf("128GiB should not RAM-cap routing: %+v", big)
	}
	if got := flashMoESlotBankEnvLine(big); !strings.Contains(got, "ZEROLLAMA_FLASH_MOE_SLOT_BANK=") {
		t.Fatalf("env line %q", got)
	}
	packed := flashMoEResolveRowFrom(discover.FlashMoEInventoryEntry{
		ExpertCount: 256, ExpertWeightBytes: 256 << 20,
	}, 128)
	if packed.SlotBankBytes != int64(packed.RecommendSlotBank)<<20 {
		t.Fatalf("bank bytes %+v", packed)
	}
	if packed.StickyMissRate <= 0 || packed.StickyMissRate > 0.15+1e-9 {
		t.Fatalf("sticky miss %+v", packed)
	}
	tight := flashMoEResolveRowFrom(discover.FlashMoEInventoryEntry{ExpertCount: 256}, 8)
	if tight.RamCapSlots != 16 || tight.RecommendSlotBank > 16 {
		t.Fatalf("8GiB %+v", tight)
	}
}

func TestGroupFlashMoERowsByGGUF(t *testing.T) {
	blob := "/blobs/sha256-same"
	rows := []flashMoEResolveRow{
		{FlashMoEInventoryEntry: discover.FlashMoEInventoryEntry{Tag: "ane-ffn-lab-shexp:latest", GGUFPath: blob}},
		{FlashMoEInventoryEntry: discover.FlashMoEInventoryEntry{Tag: "qwen3.6-mtp:latest", GGUFPath: blob}},
		{FlashMoEInventoryEntry: discover.FlashMoEInventoryEntry{Tag: "other:latest", GGUFPath: "/blobs/other"}},
	}
	g := groupFlashMoERowsByGGUF(rows)
	if len(g) != 2 {
		t.Fatalf("groups=%d", len(g))
	}
	if len(g[0].tags) != 2 || g[0].tags[0] != "ane-ffn-lab-shexp:latest" {
		t.Fatalf("aliases %+v", g[0].tags)
	}
}
