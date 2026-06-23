package envconfig

import (
	"os"
	"path/filepath"
	"testing"
)

func TestANERepoDefault(t *testing.T) {
	t.Setenv("ANE_REPO", "")
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	want := filepath.Join(home, "Sites", "inference", "ane")
	if got := ANERepo(); got != want {
		t.Fatalf("ANERepo() = %q want %q", got, want)
	}
}

func TestANERepoEnv(t *testing.T) {
	t.Setenv("ANE_REPO", "/tmp/custom-ane")
	if got := ANERepo(); got != "/tmp/custom-ane" {
		t.Fatalf("ANERepo() = %q", got)
	}
}

func TestFlashMoERepoDefault(t *testing.T) {
	t.Setenv("FLASH_MOE_REPO", "")
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	want := filepath.Join(home, "Sites", "inference", "anemll-flash-llama.cpp")
	if got := FlashMoERepo(); got != want {
		t.Fatalf("FlashMoERepo() = %q want %q", got, want)
	}
}

func TestFlashMoERepoEnv(t *testing.T) {
	t.Setenv("FLASH_MOE_REPO", "/tmp/custom-flash-moe")
	if got := FlashMoERepo(); got != "/tmp/custom-flash-moe" {
		t.Fatalf("FlashMoERepo() = %q", got)
	}
}
