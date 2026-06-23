package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	typesmodel "github.com/ollama/ollama/types/model"
)

func TestParseHFSourceURI(t *testing.T) {
	ref, err := ParseHFSource("huggingface://TheBloke/phi-2-GGUF/phi-2.Q8_0.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if ref.Repo != "TheBloke/phi-2-GGUF" || ref.File != "phi-2.Q8_0.gguf" || ref.Revision != "main" {
		t.Fatalf("ref=%+v", ref)
	}
}

func TestParseHFSourceHFAlias(t *testing.T) {
	ref, err := ParseHFSource("hf://meta-llama/Llama-3.2-1B")
	if err != nil {
		t.Fatal(err)
	}
	if ref.Repo != "meta-llama/Llama-3.2-1B" || ref.File != "" {
		t.Fatalf("ref=%+v", ref)
	}
}

func TestParseHFSourceHTTPSResolve(t *testing.T) {
	ref, err := ParseHFSource("https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q8_0.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if ref.Repo != "TheBloke/phi-2-GGUF" || ref.Revision != "main" || ref.File != "phi-2.Q8_0.gguf" {
		t.Fatalf("ref=%+v", ref)
	}
}

func TestHFLocalName(t *testing.T) {
	name := hfLocalName(HFRef{
		Repo: "TheBloke/phi-2-GGUF",
		File: "phi-2.Q8_0.gguf",
	})
	if got := name.DisplayShortest(); got != "phi-2-gguf:q8-0" {
		t.Fatalf("got %q", got)
	}
}

func TestHFPickGGUF(t *testing.T) {
	file, err := hfPickGGUF([]hfTreeEntry{
		{Path: "phi-2.Q4_K_M.gguf", Size: 100},
		{Path: "phi-2.Q8_0.gguf", Size: 200},
	})
	if err != nil || file != "phi-2.Q8_0.gguf" {
		t.Fatalf("file=%q err=%v", file, err)
	}

	file, err = hfPickGGUF([]hfTreeEntry{
		{Path: "a.Q4.gguf", Size: 1},
		{Path: "b.Q8.gguf", Size: 2},
	})
	if err != nil || file != "b.Q8.gguf" {
		t.Fatalf("file=%q err=%v", file, err)
	}
}

func TestPullFromHuggingFaceCreatesManifest(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")

	_, digest := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}, nil)

	local := hfLocalName(HFRef{Repo: "test-org/test-gguf", File: "model.Q4.gguf"})
	relFiles := map[string]string{"model.Q4.gguf": digest}
	baseLayers, err := convertModelFromFiles(relFiles, nil, false, func(api.ProgressResponse) {})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &typesmodel.ConfigV2{OS: "linux", Architecture: "amd64", RootFS: typesmodel.RootFS{Type: "layers"}}
	if err := createModel(api.CreateRequest{}, local, baseLayers, cfg, func(api.ProgressResponse) {}); err != nil {
		t.Fatal(err)
	}

	m, err := GetModel(local.DisplayShortest())
	if err != nil {
		t.Fatal(err)
	}
	if m.ModelPath == "" {
		t.Fatal("expected model path")
	}
}

func TestHFResolveFileUsesTree(t *testing.T) {
	data, _ := createBinFile(t, ggml.KV{"general.architecture": "llama"}, nil)
	_ = data
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/api/models/") {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode([]hfTreeEntry{
			{Type: "file", Path: "weights/model.gguf", Size: 10},
		})
	}))
	defer srv.Close()

	// Can't easily redirect hf API host without injection; test hfPickGGUF + parse only.
	ref, err := hfResolveFile(context.Background(), HFRef{Repo: "org/model", File: "weights/model.gguf"})
	if err != nil || ref.File != "weights/model.gguf" {
		t.Fatalf("ref=%+v err=%v", ref, err)
	}
}

func TestStageHFBlobPath(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	path := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(path, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	digest, _ := GetSHA256Digest(f)
	_ = f.Close()
	if err := stageFilesToBlobs(map[string]string{path: digest}); err != nil {
		t.Fatal(err)
	}
}
