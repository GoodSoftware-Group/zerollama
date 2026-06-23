package llm

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestAppendFlashMoEArgsInactive(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE", "")
	got := appendFlashMoEArgs(nil, api.Options{})
	if len(got) != 0 {
		t.Fatalf("appendFlashMoEArgs = %v, want empty", got)
	}
}

func TestAppendFlashMoEArgsFromEnv(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE", "1")
	t.Setenv("ZEROLLAMA_FLASH_MOE_SIDECAR", "/tmp/flash-sidecar")
	t.Setenv("ZEROLLAMA_FLASH_MOE_SLOT_BANK", "32")
	t.Setenv("ZEROLLAMA_FLASH_MOE_TOPK", "4")
	t.Setenv("ZEROLLAMA_FLASH_MOE_PREFETCH", "1")

	got := appendFlashMoEArgs([]string{"-b", "64", "-ub", "64"}, api.Options{})
	want := []string{
		"-b", "64", "-ub", "1",
		"--moe-mode", "slot-bank",
		"--moe-sidecar", "/tmp/flash-sidecar",
		"--moe-slot-bank", "32",
		"--moe-topk", "4",
		"--moe-prefetch-temporal",
		"-fit", "on",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("appendFlashMoEArgs = %v, want %v", got, want)
	}
}

func TestAppendFlashMoEArgsManifestOverridesEnv(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE", "")
	prefetch := true
	got := appendFlashMoEArgs(nil, api.Options{
		Runner: api.Runner{
			MoeSidecar:          "/models/flash/qwen35",
			MoeMode:             "slot-bank",
			MoeSlotBank:         16,
			MoeTopK:             4,
			MoePrefetchTemporal: &prefetch,
		},
	})
	want := []string{
		"--moe-mode", "slot-bank",
		"--moe-sidecar", "/models/flash/qwen35",
		"--moe-slot-bank", "16",
		"--moe-topk", "4",
		"--moe-prefetch-temporal",
		"-fit", "on",
		"-ub", "1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("appendFlashMoEArgs = %v, want %v", got, want)
	}
}

func TestAppendFlashMoEArgsPrefetchFalseWithoutSidecarInactive(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE", "")
	prefetch := false
	got := appendFlashMoEArgs(nil, api.Options{
		Runner: api.Runner{
			MoePrefetchTemporal: &prefetch,
		},
	})
	if len(got) != 0 {
		t.Fatalf("appendFlashMoEArgs = %v, want empty without sidecar", got)
	}
}

func TestAppendFlashMoEArgsPrefetchFalseWithSidecar(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLASH_MOE", "")
	prefetch := false
	got := appendFlashMoEArgs(nil, api.Options{
		Runner: api.Runner{
			MoeSidecar:          "/models/flash/qwen35",
			MoePrefetchTemporal: &prefetch,
		},
	})
	if containsFlag(got, "--moe-prefetch-temporal") {
		t.Fatalf("did not expect prefetch flag: %v", got)
	}
	if got[len(got)-2] != "-ub" || got[len(got)-1] != "1" {
		t.Fatalf("expected -ub 1 at end, got %v", got[len(got)-2:])
	}
}

func TestSetLlamaServerUbatchReplace(t *testing.T) {
	got := setLlamaServerUbatch([]string{"-b", "64", "-ub", "64"}, 1)
	want := []string{"-b", "64", "-ub", "1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("setLlamaServerUbatch = %v", got)
	}
}

func containsFlag(params []string, flag string) bool {
	for _, p := range params {
		if p == flag {
			return true
		}
	}
	return false
}
