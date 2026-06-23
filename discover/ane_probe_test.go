package discover

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindANEProbeBinEmptyWhenMissing(t *testing.T) {
	t.Setenv("ZEROLLAMA_ANE_PROBE", "")
	// Do not assume probe is built in CI; just ensure lookup does not panic.
	_ = FindANEProbeBin()
}

func TestANERepoPathDefault(t *testing.T) {
	t.Setenv("ANE_REPO", "")
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	want := filepath.Join(home, "Sites", "inference", "ane")
	if got := ANERepoPath(); got != want {
		t.Fatalf("ANERepoPath() = %q want %q", got, want)
	}
}

func TestANERepoPathEnv(t *testing.T) {
	t.Setenv("ANE_REPO", "/tmp/custom-ane")
	if got := ANERepoPath(); got != "/tmp/custom-ane" {
		t.Fatalf("ANERepoPath() = %q", got)
	}
}
