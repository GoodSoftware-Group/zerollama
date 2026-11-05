package server

import "testing"

func TestParseGGUFSplitName(t *testing.T) {
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
		idx, n, ok := parseGGUFSplitName(tt.base)
		if ok != tt.ok || idx != tt.idx || n != tt.n {
			t.Fatalf("%s: got idx=%d n=%d ok=%v want idx=%d n=%d ok=%v",
				tt.base, idx, n, ok, tt.idx, tt.n, tt.ok)
		}
	}
}
