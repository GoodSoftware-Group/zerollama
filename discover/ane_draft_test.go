package discover

import "testing"

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
