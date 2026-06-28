package discover

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestANEDraftWeightCachePath(t *testing.T) {
	got := aneDraftWeightCachePath("/tmp/drafter-2b.gguf", 256, "blk.0.ffn_gate.weight")
	if !strings.Contains(got, "drafter-2b-256-blk_0_ffn_gate_weight.v3.weight.bin") {
		t.Fatalf("unexpected cache path: %s", got)
	}
}

func TestMaterializeANEDraftWeightFileUsesCache(t *testing.T) {
	if runtimeGOOS := os.Getenv("GOOS"); runtimeGOOS == "windows" {
		t.Skip("darwin-focused")
	}
	sidecar := filepath.Join(os.Getenv("HOME"), ".cache", "zerollama", "eliza-1", "bundles", "2b", "dflash", "drafter-2b.gguf")
	if st, err := os.Stat(sidecar); err != nil || st.IsDir() {
		t.Skip("drafter sidecar not present")
	}

	dir := t.TempDir()
	t.Setenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_CACHE", dir)

	entry := ANEDraftEntry{
		Tag:                 "eliza-1-2b-dflash:latest",
		DraftGGUF:           sidecar,
		DraftSidecarPresent: true,
		EmbeddingLength:     2048,
		ProxyChannels:       256,
		ProxySpatial:        16,
	}

	path1, cached1, err := MaterializeANEDraftWeightFile(entry, "")
	if err != nil {
		t.Fatal(err)
	}
	if cached1 {
		t.Fatal("first materialize should write cache")
	}
	if _, err := os.Stat(path1); err != nil {
		t.Fatal(err)
	}

	path2, cached2, err := MaterializeANEDraftWeightFile(entry, "")
	if err != nil {
		t.Fatal(err)
	}
	if !cached2 {
		t.Fatal("second materialize should hit cache")
	}
	if path1 != path2 {
		t.Fatalf("cache paths differ: %q vs %q", path1, path2)
	}
}

func TestMaterializeANEDraftWeightFileMissingSidecar(t *testing.T) {
	entry := ANEDraftEntry{
		Tag:                 "test-dflash:latest",
		DraftGGUF:           "/no/such/drafter.gguf",
		DraftSidecarPresent: false,
		ProxyChannels:       64,
	}
	_, _, err := MaterializeANEDraftWeightFile(entry, "")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestMaterializeANEDraftWeightBundleConv2(t *testing.T) {
	sidecar := filepath.Join(os.Getenv("HOME"), ".cache", "zerollama", "eliza-1", "bundles", "2b", "dflash", "drafter-2b.gguf")
	if st, err := os.Stat(sidecar); err != nil || st.IsDir() {
		t.Skip("drafter sidecar not present")
	}
	entry := ANEDraftEntry{
		Tag:                 "eliza-1-2b-dflash:latest",
		DraftGGUF:           sidecar,
		DraftSidecarPresent: true,
		EmbeddingLength:     2048,
		ProxyChannels:       256,
		ProxySpatial:        16,
		SpecType:            "dflash",
	}
	manifest, _, err := MaterializeANEDraftWeightBundle(entry)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.ConvWeightPath() == "" {
		t.Fatal("missing conv weight")
	}
	if manifest.Conv2WeightPath() == "" {
		t.Fatal("expected conv2 weight from ffn_up sidecar")
	}
	if manifest.Version < 2 {
		t.Fatalf("manifest version = %d, want >= 2", manifest.Version)
	}
}
