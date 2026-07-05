package discover

import (
	"strings"
	"testing"
)

func TestSelectANEDraftModelPreferred(t *testing.T) {
	entries := []ANEDraftEntry{
		{Tag: "eliza-1-2b:latest", Name: "registry.ollama.ai/library/eliza-1-2b:latest"},
		{Tag: "eliza-1-2b-dflash:latest", Name: "registry.ollama.ai/library/eliza-1-2b-dflash:latest"},
	}
	got, ok := SelectANEDraftModel(entries, "dflash")
	if !ok || got.Tag != "eliza-1-2b-dflash:latest" {
		t.Fatalf("SelectANEDraftModel = %+v ok=%v", got, ok)
	}
}

func TestFindANEDraftSidecarPathEmpty(t *testing.T) {
	if got := FindANEDraftSidecarPath(""); got != "" {
		t.Fatalf("got %q", got)
	}
}

func TestDflashParentShort(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"eliza-1-2b-dflash", "eliza-1-2b"},
		{"eliza-1-27b-256k-dflash", "eliza-1-27b-256k"},
		{"eliza-1-2b", ""},
	}
	for _, tc := range tests {
		if got := dflashParentShort(tc.in); got != tc.want {
			t.Fatalf("dflashParentShort(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestElizaDflashInventoryUsesParentBase(t *testing.T) {
	entries, err := ListANEDraftInventory()
	if err != nil {
		t.Skip(err)
	}
	entry, ok := SelectANEDraftModel(entries, "eliza-1-2b-dflash")
	if !ok {
		t.Skip("eliza-1-2b-dflash not in inventory")
	}
	if !strings.Contains(entry.BaseGGUF, "sha256-a511452ec932613d6b26b4fa24488fd431eb61eac69321460447d475edc221e2") {
		t.Fatalf("base_gguf = %q, want eliza-1-2b parent blob (a511452e…)", entry.BaseGGUF)
	}
}
