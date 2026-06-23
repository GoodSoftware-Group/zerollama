package server

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func writeRepairTestManifest(t *testing.T, name model.Name, kv ggml.KV, params map[string]any, cfg *model.ConfigV2) {
	t.Helper()
	_, digest := createBinFile(t, kv, nil)
	modelLayer, err := manifest.NewLayerFromLayer(digest, "application/vnd.ollama.image.model", "test.gguf")
	if err != nil {
		t.Fatal(err)
	}
	layers := []manifest.Layer{modelLayer}
	if len(params) > 0 {
		layers, err = setParameters(layers, params)
		if err != nil {
			t.Fatal(err)
		}
	}
	config := model.ConfigV2{}
	if cfg != nil {
		config = *cfg
	}
	configLayer, err := createConfigLayer(layers, config)
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(name, *configLayer, layers); err != nil {
		t.Fatal(err)
	}
}

func TestRepairCapsNumCtxDryRun(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	name := model.ParseName("repair-cap")
	writeRepairTestManifest(t, name, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(131072),
	}, map[string]any{"num_ctx": 131072}, nil)

	r, err := RepairModel(name.String(), RepairOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Skipped {
		t.Fatalf("skipped: %s", r.Reason)
	}
	if r.Written {
		t.Fatal("dry-run should not write")
	}
	found := false
	for _, c := range r.Changes {
		if c.Field == "params.num_ctx" && c.From == "131072" && c.To == "8192" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected num_ctx cap change, got %+v", r.Changes)
	}
}

func TestRepairCapsNumCtxWrite(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	name := model.ParseName("repair-write")
	writeRepairTestManifest(t, name, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(131072),
	}, map[string]any{"num_ctx": 131072}, nil)

	r, err := RepairModel(name.String(), RepairOptions{Write: true})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Written {
		t.Fatal("expected write")
	}

	m, err := GetModel(name.String())
	if err != nil {
		t.Fatal(err)
	}
	n, ok := m.Options["num_ctx"].(float64)
	if !ok || int(n) != defaultManifestNumCtxCap {
		t.Fatalf("num_ctx=%v want %d", m.Options["num_ctx"], defaultManifestNumCtxCap)
	}
}

func TestRepairFillsParser(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	name := model.ParseName("repair-parser")
	writeRepairTestManifest(t, name, ggml.KV{
		"general.architecture":    "gemma4",
		"general.parameter_count": uint64(4_300_000_000),
	}, nil, nil)

	r, err := RepairModel(name.String(), RepairOptions{Write: true})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Written {
		t.Fatalf("expected write, changes=%+v", r.Changes)
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	path, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var cfg model.ConfigV2
	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Parser != "gemma4-no-thinking" {
		t.Fatalf("parser=%q", cfg.Parser)
	}
	if cfg.Renderer != gemma4RendererLegacy {
		t.Fatalf("renderer=%q", cfg.Renderer)
	}
}

func TestRepairSkipsNoGGUF(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("mlx-only")
	cfg := model.ConfigV2{ModelFormat: "safetensors"}
	layers := []manifest.Layer{}
	configLayer, err := createConfigLayer(layers, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(name, *configLayer, layers); err != nil {
		t.Fatal(err)
	}

	r, err := RepairModel(name.String(), RepairOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Skipped || r.Reason == "" {
		t.Fatalf("expected skip, got %+v", r)
	}
}

func TestRepairSkipsWhenGuessingDisabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "1")
	name := model.ParseName("repair-disabled")
	writeRepairTestManifest(t, name, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}, map[string]any{"num_ctx": 131072}, nil)

	r, err := RepairModel(name.String(), RepairOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Skipped {
		t.Fatalf("expected skip, got %+v", r)
	}
}

func TestRepairNoChangesOK(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	name := model.ParseName("repair-ok")
	writeRepairTestManifest(t, name, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}, map[string]any{"num_ctx": 4096}, &model.ConfigV2{
		ModelFamily:   "llama",
		ModelFamilies: []string{"llama"},
		ContextLen:    4096,
	})

	r, err := RepairModel(name.String(), RepairOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Skipped || len(r.Changes) > 0 {
		t.Fatalf("expected no changes, got %+v", r)
	}
}
