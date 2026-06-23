package discover

import "testing"

func TestDraftMILBlockers(t *testing.T) {
	b := draftMILBlockers(false, true, true)
	if len(b) != 1 || b[0] != "eagle3 drafter GGUF missing" {
		t.Fatalf("sidecar blocker: %+v", b)
	}
	b = draftMILBlockers(true, false, true)
	if len(b) != 1 {
		t.Fatalf("lab bins blocker: %+v", b)
	}
	if len(draftMILBlockers(true, true, true)) != 0 {
		t.Fatal("expected no blockers")
	}
}
