package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildMistral3PaddedCompletionPromptTokens_splice(t *testing.T) {
	rendered := "<s>[INST] old [/INST]assistant"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "old",
		PaddedInputIDs: []int{10, 12, 10, 13},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeMistral3Img}
	got, ok := BuildMistral3PaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("<s>[INST] ")
	if got[prefixLen] != 10 || got[prefixLen+3] != 13 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+4])
	}
}

func TestPaddedLayoutConsumePlan_mistral3(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("", []string{"mistral3"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeMistral3Img {
		t.Fatalf("plan=%+v", plan)
	}
}
