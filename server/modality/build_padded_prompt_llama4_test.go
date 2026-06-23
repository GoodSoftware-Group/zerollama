package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildLlama4PaddedCompletionPromptTokens_splice(t *testing.T) {
	rendered := "<|header_start|>user<|header_end|>\n\nold<|eot|>"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "old",
		PaddedInputIDs: []int{10, 20, 30},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeLlama4Img}
	got, ok := BuildLlama4PaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("<|header_start|>user<|header_end|>\n\n")
	if got[prefixLen] != 10 || got[prefixLen+2] != 30 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+3])
	}
}

func TestPaddedLayoutConsumePlan_llama4(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("", []string{"llama4"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeLlama4Img {
		t.Fatalf("plan=%+v", plan)
	}
}
