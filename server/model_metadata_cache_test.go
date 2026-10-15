package server

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func writeTestGGUF(t *testing.T, path string, kv ggml.KV) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	base := ggml.KV{"general.architecture": "llama", "llama.context_length": uint32(4096)}
	for k, v := range kv {
		base[k] = v
	}
	if err := ggml.WriteGGUF(f, base, nil); err != nil {
		t.Fatal(err)
	}
}

// TestModelFileCachesRevalidate pins the per-request header caches (Finding 22): repeat
// calls reuse the decode, and rewriting the file (size / mtime change — LM Studio repair,
// an external --link GGUF) is picked up instead of served stale.
func TestModelFileCachesRevalidate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.gguf")
	writeTestGGUF(t, path, ggml.KV{"llama.context_length": uint32(4096)})

	a, err := llm.LoadModelMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := llm.LoadModelMetadata(path)
	if a != b {
		t.Fatal("second LoadModelMetadata decoded again; want the cached header")
	}
	m := &Model{ModelPath: path, Name: "m"}
	if caps := m.Capabilities(); slices.Contains(caps, model.CapabilityVision) {
		t.Fatalf("caps before rewrite = %v", caps)
	}

	writeTestGGUF(t, path, ggml.KV{"llama.context_length": uint32(32768), "vision.block_count": uint32(2), "llama.decision.type": "clef"})
	future := time.Now().Add(time.Minute) // mtime granularity: force a visible change
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}

	c, err := llm.LoadModelMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := c.KV().ContextLength(); got != 32768 {
		t.Fatalf("context length after rewrite = %d, want 32768 (stale cache)", got)
	}
	if caps := m.Capabilities(); !slices.Contains(caps, model.CapabilityVision) || !slices.Contains(caps, model.CapabilityDecision) {
		t.Fatalf("caps after rewrite = %v, want vision + decision (stale cache)", caps)
	}
}
