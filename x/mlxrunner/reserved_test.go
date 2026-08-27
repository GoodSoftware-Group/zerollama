package mlxrunner

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/tokenizer"
)

func TestBanReservedSampleText(t *testing.T) {
	if !banReservedSampleText("<|fim_hole|>") || !banReservedSampleText("<|reserved_token_0|>") {
		t.Fatal("expected FIM/reserved banned")
	}
	if banReservedSampleText("<think>") || banReservedSampleText("</think>") ||
		banReservedSampleText("<tool_call>") || banReservedSampleText("<|im_end|>") {
		t.Fatal("think/tool/im_end must stay sampleable")
	}
}

func TestReservedSampleBanIDs(t *testing.T) {
	vocab := map[string]int32{"a": 0, "b": 1, "<|fim_hole|>": 2, "<think>": 3, "<|endoftext|>": 4}
	model := map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}}
	data, err := json.Marshal(map[string]any{
		"model": model,
		"added_tokens": []map[string]any{
			{"id": 2, "content": "<|fim_hole|>", "special": true},
			{"id": 3, "content": "<think>", "special": true},
			{"id": 4, "content": "<|endoftext|>", "special": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	gen, err := json.Marshal(map[string]any{"eos_token_id": 4})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(data, &tokenizer.TokenizerConfig{GenerationConfigJSON: gen})
	if err != nil {
		t.Fatal(err)
	}
	got := reservedSampleBanIDs(tok)
	if !slices.Equal(got, []int32{2}) {
		t.Fatalf("banned %v, want [2]", got)
	}
}

func TestSuppressReservedEnabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_SUPPRESS_RESERVED", "")
	if !suppressReservedEnabled() {
		t.Fatal("default on")
	}
	t.Setenv("ZEROLLAMA_MLX_SUPPRESS_RESERVED", "0")
	if suppressReservedEnabled() {
		t.Fatal("0 disables")
	}
}
