package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/internal/lmstudio"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestManifestHasMissingBlobs(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("missing-blob")
	writeRepairTestManifest(t, name, ggml.KV{"general.architecture": "llama"}, nil, nil)

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	if manifestHasMissingBlobs(mf) {
		t.Fatal("expected healthy manifest")
	}

	path, err := manifest.BlobsPath(mf.Layers[0].Digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if !manifestHasMissingBlobs(mf) {
		t.Fatal("expected missing blob")
	}
}

func TestNeedsLMStudioSync(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("library/sync-test:latest")
	if !needsLMStudioSync(name) {
		t.Fatal("missing manifest should need sync")
	}

	writeRepairTestManifest(t, name, ggml.KV{"general.architecture": "llama"}, nil, nil)
	if needsLMStudioSync(name) {
		t.Fatal("healthy manifest should not need sync")
	}
}

func TestSyncLMStudioModels_ImportsFromCache(t *testing.T) {
	modelsDir := t.TempDir()
	cacheRoot := t.TempDir()
	modelDir := filepath.Join(cacheRoot, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}

	ggufPath := filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf")
	srcPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}, nil)
	if err := copyTestGGUF(srcPath, ggufPath); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_MODELS", modelsDir)
	t.Setenv("OLLAMA_LMSTUDIO_MODELS", cacheRoot)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")

	name := model.ParseName("lmstudio-community/gemma-4-31b-it:q8_0")
	if err := SyncLMStudioModels(t.Context()); err != nil {
		t.Fatal(err)
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatalf("manifest missing after sync: %v", err)
	}
	if manifestHasMissingBlobs(mf) {
		t.Fatal("sync left missing blobs")
	}
}

func TestSyncLMStudioModels_RepairsMissingBlob(t *testing.T) {
	modelsDir := t.TempDir()
	cacheRoot := t.TempDir()
	modelDir := filepath.Join(cacheRoot, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}

	ggufPath := filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf")
	srcPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}, nil)
	if err := copyTestGGUF(srcPath, ggufPath); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_MODELS", modelsDir)
	t.Setenv("OLLAMA_LMSTUDIO_MODELS", cacheRoot)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")

	name := model.ParseName("lmstudio-community/gemma-4-31b-it:q8_0")
	writeRepairTestManifest(t, name, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}, nil, nil)

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	blobPath, err := manifest.BlobsPath(mf.Layers[0].Digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(blobPath); err != nil {
		t.Fatal(err)
	}

	if err := SyncLMStudioModels(t.Context()); err != nil {
		t.Fatal(err)
	}

	mf, err = manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	if manifestHasMissingBlobs(mf) {
		t.Fatal("sync did not repair missing blob")
	}
}

func TestSyncLMStudioModels_Disabled(t *testing.T) {
	modelsDir := t.TempDir()
	cacheRoot := t.TempDir()
	modelDir := filepath.Join(cacheRoot, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_MODELS", modelsDir)
	t.Setenv("OLLAMA_LMSTUDIO_MODELS", cacheRoot)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "false")

	if err := SyncLMStudioModels(t.Context()); err != nil {
		t.Fatal(err)
	}

	name := model.ParseName("lmstudio-community/gemma-4-31b-it:q8_0")
	if _, err := manifest.ParseNamedManifest(name); !os.IsNotExist(err) {
		t.Fatalf("expected no manifest when import disabled, got err=%v", err)
	}
}

func TestLMStudioEntrySyncable(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "Hermes-4-70B-MLX-8bit")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "model-00001-of-00002.safetensors"} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_MODELS", t.TempDir())
	restore := lmstudio.SetModelsFreeBytesHook(func(string) (int64, error) { return 0, nil })
	t.Cleanup(restore)

	entry := lmstudio.Entry{Name: "lmstudio-community/hermes-4-70b:8bit", Dir: modelDir, Format: "safetensors"}
	if lmStudioEntrySyncable(entry) {
		t.Fatal("MLX entry should not sync without disk")
	}

	t.Setenv("OLLAMA_LMSTUDIO_LIST_ALL", "true")
	if !lmStudioEntrySyncable(entry) {
		t.Fatal("LIST_ALL should allow sync attempt")
	}
}

func copyTestGGUF(src, dest string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dest, data, 0o644)
}
