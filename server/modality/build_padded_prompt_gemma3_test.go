package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildGemma3PaddedCompletionPromptTokens_splice(t *testing.T) {
	rendered := "<start_of_turn>user\nold<end_of_turn>"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "old",
		PaddedInputIDs: []int{10, 20, 30},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeGemma3Img}
	got, ok := BuildGemma3PaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("<start_of_turn>user\n")
	if got[prefixLen] != 10 || got[prefixLen+2] != 30 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+3])
	}
}

func TestPaddedLayoutConsumePlan_gemma3(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("gemma3", []string{"gemma3"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeGemma3Img {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestPaddedLayoutConsumePlan_gemma3nDeferred(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("", []string{"gemma3n"}, msgs, true)
	if plan.Mode != PaddedLayoutConsumeDeferredNonQwen3V {
		t.Fatalf("gemma3n text-only should defer: plan=%+v", plan)
	}
}
