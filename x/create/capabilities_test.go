package create

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestSourceCapabilitiesDecisionHeads(t *testing.T) {
	t.Run("laya", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["LayaForDecision"]}`), 0o644); err != nil {
			t.Fatal(err)
		}
		caps, err := SourceCapabilities(dir)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(caps, string(model.CapabilityDecision)) {
			t.Fatalf("caps=%v want decision", caps)
		}
	})

	t.Run("clef vision", func(t *testing.T) {
		dir := t.TempDir()
		cfg := `{"architectures":["ClefForDecision"],"vision_config":{"hidden_size":16}}`
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
		caps, err := SourceCapabilities(dir)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(caps, string(model.CapabilityDecision)) || !slices.Contains(caps, "vision") {
			t.Fatalf("caps=%v want decision+vision", caps)
		}
	})

	t.Run("nimble head layout", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "decision_head_config.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
		caps, err := SourceCapabilities(dir)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(caps, string(model.CapabilityDecision)) || !slices.Contains(caps, string(model.CapabilityCompletion)) {
			t.Fatalf("caps=%v want completion+decision", caps)
		}
	})

	t.Run("plain qwen no head", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		caps, err := SourceCapabilities(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(caps) != 0 {
			t.Fatalf("caps=%v want empty", caps)
		}
	})
}

func TestHasDecisionHeadLayout(t *testing.T) {
	dir := t.TempDir()
	if hasDecisionHeadLayout(dir) {
		t.Fatal("empty dir should be false")
	}
	if err := os.WriteFile(filepath.Join(dir, "tev1_config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if !hasDecisionHeadLayout(dir) {
		t.Fatal("tev1_config.json should detect decision head")
	}
}
