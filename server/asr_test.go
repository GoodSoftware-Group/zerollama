package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestASRTranscriptText(t *testing.T) {
	text, usage, err := asrTranscriptText([]byte(`{"type":"transcript.text.done","text":" hello ","usage":{"type":"tokens","input_tokens":4,"output_tokens":1,"total_tokens":5}}`))
	if err != nil || text != "hello" || usage == nil || usage.InputTokens != 4 || usage.TotalTokens != 5 {
		t.Fatalf("text=%q usage=%+v err=%v", text, usage, err)
	}
	if _, _, err := asrTranscriptText([]byte(`{"type":"transcript.text.done"}`)); err == nil {
		t.Fatal("expected empty transcript error")
	}
	if _, _, err := asrTranscriptText([]byte(`not-json`)); err == nil {
		t.Fatal("expected json error")
	}
}

func TestModelUsesLlamaASR(t *testing.T) {
	if !modelUsesLlamaASR(&Model{Config: model.ConfigV2{ModelFamily: "parakeet", ModelFamilies: []string{"parakeet"}}}) {
		t.Fatal("family")
	}
	if !modelUsesLlamaASR(&Model{Config: model.ConfigV2{ModalityBackends: map[string]string{model.ModalityTranscribe: model.BackendLlamaASR}}}) {
		t.Fatal("backend")
	}
	if modelUsesLlamaASR(&Model{Config: model.ConfigV2{ModelFamily: "qwen35"}}) {
		t.Fatal("chat family")
	}
	if modelUsesLlamaASR(nil) {
		t.Fatal("nil")
	}
}
