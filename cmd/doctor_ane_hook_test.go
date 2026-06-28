package cmd

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestDoctorANESourceMarkersPresentInCanonical(t *testing.T) {
	repo := doctorRepoRoot()
	root := filepath.Join(repo, "tools", "ane-patches", "canonical", "common")
	specPath := filepath.Join(repo, "llama", "llama.cpp", "common", "speculative.cpp")
	if _, err := os.Stat(specPath); err != nil {
		t.Skip("in-tree speculative.cpp missing")
	}
	specOK, filesOK, detail := doctorANESourceMarkers(filepath.Join(repo, "llama", "llama.cpp"))
	if !filesOK {
		t.Fatalf("in-tree files: %s", detail)
	}
	if !specOK {
		t.Fatalf("in-tree speculative: %s", detail)
	}
	_ = root
	for _, f := range []string{"ane_draft_hook.cpp", "ane_draft_session.mm"} {
		if _, err := os.Stat(filepath.Join(root, f)); err != nil {
			t.Fatalf("canonical missing %s: %v", f, err)
		}
	}
}

func TestDoctorCheckANEDraftHookNonDarwinSkips(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("non-darwin only")
	}
	c := doctorCheckANEDraftHook(".")
	if c.Status != "ok" {
		t.Fatalf("status=%q detail=%q", c.Status, c.Detail)
	}
}
