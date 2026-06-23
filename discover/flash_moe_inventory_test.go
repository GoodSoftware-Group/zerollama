package discover

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestIsMoEGGUF(t *testing.T) {
	kv := ggml.KV{
		"general.architecture": "qwen35moe",
		"qwen35moe.expert_count": uint32(64),
	}
	if !isMoEGGUF(kv, "") {
		t.Fatal("qwen35moe with experts should count as MoE")
	}
	if isMoEGGUF(ggml.KV{"general.architecture": "llama"}, "llama") {
		t.Fatal("dense llama should not count as MoE")
	}
	if !isMoEGGUF(ggml.KV{}, "granite3-moe") {
		t.Fatal("family containing moe should count")
	}
}

func TestFlashMoESidecarFromParams(t *testing.T) {
	got := flashMoESidecarFromParams(map[string]any{"moe_sidecar": " /tmp/sidecar "})
	if got != "/tmp/sidecar" {
		t.Fatalf("got %q", got)
	}
}

func TestFlashMoESidecarReady(t *testing.T) {
	dir := t.TempDir()
	if flashMoESidecarReady(dir) {
		t.Fatal("empty dir should not be ready")
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !flashMoESidecarReady(dir) {
		t.Fatal("dir with manifest.json should be ready")
	}
}

func TestSelectFlashMoEModelPreferred(t *testing.T) {
	entries := []FlashMoEInventoryEntry{
		{Tag: "small:latest", Name: "registry.ollama.ai/library/small:latest"},
		{Tag: "qwen35:latest", Name: "registry.ollama.ai/library/qwen35:latest", SidecarReady: true},
	}
	got, ok := SelectFlashMoEModel(entries, "qwen35")
	if !ok || got.Tag != "qwen35:latest" {
		t.Fatalf("SelectFlashMoEModel = %+v ok=%v", got, ok)
	}
	got, ok = SelectFlashMoEModel(entries, "")
	if !ok || !got.SidecarReady {
		t.Fatalf("default pick should prefer sidecar-ready, got %+v", got)
	}
}

func TestFirstNonEmpty(t *testing.T) {
	if got := firstNonEmpty("", "  a ", "b"); got != "a" {
		t.Fatalf("got %q", got)
	}
}

func TestFlashMoEDefaultSidecarPathUsesHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	got := flashMoEDefaultSidecarPath("qwen35", "/blobs/model.gguf")
	if !strings.HasSuffix(got, filepath.Join("Models", "flash", "qwen35")) {
		t.Fatalf("got %q", got)
	}
}
