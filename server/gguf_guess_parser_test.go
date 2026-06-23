package server

import (
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestGuessParserForArchitecture(t *testing.T) {
	if got := guessParserForArchitecture("qwen35", ""); got != "qwen3.5" {
		t.Fatalf("qwen35=%q", got)
	}
	if got := guessParserForArchitecture("gptoss", ""); got != "harmony" {
		t.Fatalf("gptoss=%q", got)
	}
	if got := guessParserForArchitecture("gpt_oss", ""); got != "harmony" {
		t.Fatalf("gpt_oss=%q", got)
	}
	if got := guessParserForArchitecture("qwen3", "enable thinking"); got != "qwen3-thinking" {
		t.Fatalf("qwen3 thinking=%q", got)
	}
}

func TestGuessParserFromKV(t *testing.T) {
	kv := ggml.KV{
		"general.architecture":      "gemma4",
		"tokenizer.chat_template":   "{{ thinking }}",
	}
	if got := guessParserFromKV(kv); got != "gemma4" {
		t.Fatalf("gemma4=%q", got)
	}
}
