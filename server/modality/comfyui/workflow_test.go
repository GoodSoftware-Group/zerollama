package comfyui

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTemplate(t *testing.T, dir, name, contents string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name+".json"), []byte(contents), 0o644); err != nil {
		t.Fatalf("write template: %v", err)
	}
}

const sampleT2ITemplate = `{
  "description": "sample t2i",
  "requires": ["prompt"],
  "bindings": {
    "prompt": {"node_id": "6", "field": "text"},
    "negative_prompt": {"node_id": "7", "field": "text"},
    "seed": {"node_id": "3", "field": "seed"},
    "width": {"node_id": "5", "field": "width"},
    "height": {"node_id": "5", "field": "height"},
    "lora_name": {"node_id": "10", "field": "lora_name"}
  },
  "graph": {
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "10": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "none.safetensors"}}
  }
}`

func TestLoadTemplateDirAndRender(t *testing.T) {
	dir := t.TempDir()
	writeTemplate(t, dir, "t2i", sampleT2ITemplate)

	templates, err := LoadTemplateDir(dir)
	if err != nil {
		t.Fatalf("LoadTemplateDir: %v", err)
	}
	tmpl, ok := templates["t2i"]
	if !ok {
		t.Fatalf("expected t2i template, got %v", templates)
	}

	graph, err := tmpl.Render(Inputs{
		Prompt:   "a fox in a forest",
		Seed:     42,
		SeedSet:  true,
		Width:    1024,
		Height:   768,
		LoRAName: "style.safetensors",
	})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}

	node6 := graph["6"].(map[string]any)["inputs"].(map[string]any)
	if got := node6["text"]; got != "a fox in a forest" {
		t.Errorf("prompt: got %v", got)
	}
	node3 := graph["3"].(map[string]any)["inputs"].(map[string]any)
	if got := node3["seed"]; got != int64(42) {
		t.Errorf("seed: got %v (%T)", got, got)
	}
	node5 := graph["5"].(map[string]any)["inputs"].(map[string]any)
	if got := node5["width"]; got != int32(1024) {
		t.Errorf("width: got %v (%T)", got, got)
	}
	if got := node5["height"]; got != int32(768) {
		t.Errorf("height: got %v (%T)", got, got)
	}
	node10 := graph["10"].(map[string]any)["inputs"].(map[string]any)
	if got := node10["lora_name"]; got != "style.safetensors" {
		t.Errorf("lora_name: got %v", got)
	}

	// Original template graph must be untouched by Render (deep copy).
	origNode6 := tmpl.Graph["6"].(map[string]any)["inputs"].(map[string]any)
	if got := origNode6["text"]; got != "" {
		t.Errorf("template graph mutated: got %v", got)
	}
}

func TestRenderMissingRequiredField(t *testing.T) {
	dir := t.TempDir()
	writeTemplate(t, dir, "t2i", sampleT2ITemplate)
	templates, err := LoadTemplateDir(dir)
	if err != nil {
		t.Fatalf("LoadTemplateDir: %v", err)
	}
	if _, err := templates["t2i"].Render(Inputs{}); err == nil {
		t.Fatal("expected error for missing required prompt field")
	}
}

func TestRenderUnboundFieldIgnored(t *testing.T) {
	dir := t.TempDir()
	writeTemplate(t, dir, "t2i", sampleT2ITemplate)
	templates, err := LoadTemplateDir(dir)
	if err != nil {
		t.Fatalf("LoadTemplateDir: %v", err)
	}
	// control_image has no binding in this template; must not error.
	if _, err := templates["t2i"].Render(Inputs{Prompt: "x", ControlImage: "agent-control.png"}); err != nil {
		t.Fatalf("Render with unbound field: %v", err)
	}
}

func TestLoadTemplateDirEmpty(t *testing.T) {
	dir := t.TempDir()
	if _, err := LoadTemplateDir(dir); err == nil {
		t.Fatal("expected error for directory with no templates")
	}
}
