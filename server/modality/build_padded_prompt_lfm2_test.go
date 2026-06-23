package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildLfm2PaddedCompletionPromptTokens_splice(t *testing.T) {
	rendered := "<|im_start|>user\nold<|im_end|>"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "old",
		PaddedInputIDs: []int{10, 20, 30},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeLfm2Img}
	got, ok := BuildLfm2PaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("<|im_start|>user\n")
	if got[prefixLen] != 10 || got[prefixLen+2] != 30 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+3])
	}
}

func TestPaddedLayoutConsumePlan_lfm2(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("lfm2", []string{"lfm2"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeLfm2Img {
		t.Fatalf("plan=%+v", plan)
	}
}
