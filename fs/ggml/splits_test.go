package ggml

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestParseSplitName(t *testing.T) {
	tests := []struct {
		base   string
		idx, n int
		ok     bool
	}{
		{"Llama-4-Scout-17B-16E-Instruct-Q6_K-00001-of-00002.gguf", 1, 2, true},
		{"Llama-4-Scout-17B-16E-Instruct-Q6_K-00002-of-00002.gguf", 2, 2, true},
		{"Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf", 1, 3, true},
		{"deepseek-ai.DeepSeek-V4-Flash-0731.f16.gguf.Q2_K.gguf-00008-of-00008.gguf", 8, 8, true},
		{"gpt-oss-20b-MXFP4.gguf", 0, 0, false},
		{"not-a-split.gguf", 0, 0, false},
	}
	for _, tt := range tests {
		idx, n, ok := ParseSplitName(tt.base)
		if ok != tt.ok || idx != tt.idx || n != tt.n {
			t.Fatalf("%s: got idx=%d n=%d ok=%v want idx=%d n=%d ok=%v",
				tt.base, idx, n, ok, tt.idx, tt.n, tt.ok)
		}
	}
}

func TestSplitPaths(t *testing.T) {
	dir := t.TempDir()
	prefix := "toy-model"
	n := 3
	var first string
	for i := 1; i <= n; i++ {
		name := fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, i, n)
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		if i == 1 {
			first = p
		}
	}
	paths, err := SplitPaths(first, n)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != n {
		t.Fatalf("len=%d want %d", len(paths), n)
	}
	for i, p := range paths {
		want := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, i+1, n))
		if p != want {
			t.Fatalf("paths[%d]=%q want %q", i, p, want)
		}
	}
}

func TestMergeSplitsRealQwenShards(t *testing.T) {
	primary := os.Getenv("ZEROLLAMA_TEST_QWEN35_SPLIT")
	if primary == "" {
		primary = "/Users/ryan/.ollama/models/splits/a2cd1fb39f12c877/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
	}
	if _, err := os.Stat(primary); err != nil {
		t.Skip("qwen3.5 split shards not present")
	}

	f, err := os.Open(primary)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	meta, err := Decode(f, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(meta.Tensors().Items()); n != 0 {
		t.Fatalf("first shard alone should be metadata-only, got %d tensors", n)
	}

	merged, err := MergeSplits(primary, meta)
	if err != nil {
		t.Fatal(err)
	}
	got := len(merged.Tensors().Items())
	if got != 879 {
		t.Fatalf("merged tensors=%d want 879", got)
	}
	// Spot-check a tensor from a later shard has File set.
	var fromShard2 int
	for _, tns := range merged.Tensors().Items() {
		if tns.File != "" && tns.File != primary {
			fromShard2++
		}
	}
	if fromShard2 == 0 {
		t.Fatal("expected tensors sourced from sibling shards")
	}
}
