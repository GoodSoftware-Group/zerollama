package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildGlmocrPaddedCompletionPromptTokens_splice(t *testing.T) {
	rendered := "[gMASK]<sop><|user|>\nold<|assistant|>\n"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "old",
		PaddedInputIDs: []int{10, 20, 30},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeGlmocrImg}
	got, ok := BuildGlmocrPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("[gMASK]<sop><|user|>\n")
	if got[prefixLen] != 10 || got[prefixLen+2] != 30 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+3])
	}
}

func TestBuildGlmocrPaddedCompletionPromptTokens_multiTurn(t *testing.T) {
	rendered := "[gMASK]<sop><|user|>\nfirst<|assistant|>\n<think></think>\nok\n<|user|>\nsecond<|assistant|>\n"
	msgs := []api.Message{
		{Role: "user", Content: "first", PaddedInputIDs: []int{1, 2}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "second", PaddedInputIDs: []int{9, 8}},
	}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeGlmocrImg}
	got, ok := BuildGlmocrPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	firstStart := len("[gMASK]<sop><|user|>\n")
	if got[firstStart] != 1 || got[firstStart+1] != 2 {
		t.Fatalf("first user splice wrong: %v", got[firstStart:firstStart+2])
	}
	secondMarker := "<|user|>\nsecond"
	secondIdx := len(rendered) - len("<|assistant|>\n") - len("second")
	_ = secondMarker
	_ = secondIdx
	// Locate second padded block by searching for 9,8 pair after first assistant block.
	found := false
	for i := 0; i+1 < len(got); i++ {
		if got[i] == 9 && got[i+1] == 8 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("second user padded ids missing: %v", got)
	}
}

func TestPaddedLayoutConsumePlan_glmocr(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("glm-ocr", []string{"glmocr"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeGlmocrImg {
		t.Fatalf("plan=%+v", plan)
	}
}
