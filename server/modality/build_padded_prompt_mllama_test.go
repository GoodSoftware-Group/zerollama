package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildMllamaPaddedCompletionPromptTokens_splice(t *testing.T) {
	rendered := "<|start_header_id|>user<|end_header_id|>\n\nold<|eot_id|>"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "old",
		PaddedInputIDs: []int{10, 20, 30},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeMllamaImg}
	got, ok := BuildMllamaPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("<|start_header_id|>user<|end_header_id|>\n\n")
	if got[prefixLen] != 10 || got[prefixLen+2] != 30 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+3])
	}
}

func TestPaddedLayoutConsumePlan_mllama(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("", []string{"mllama"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeMllamaImg {
		t.Fatalf("plan=%+v", plan)
	}
}
