package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildGemma4PaddedCompletionPromptTokens_splice(t *testing.T) {
	t.Parallel()
	rendered := "<bos><|turn>user\n" + "CLIP" + "<turn|>\n<|turn>model\n"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "CLIP",
		PaddedInputIDs: []int{42, 43, 44},
	}}
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeGemma4Img}
	got, ok := BuildGemma4PaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	prefixLen := len("<bos><|turn>user\n")
	if got[prefixLen] != 42 || got[prefixLen+2] != 44 {
		t.Fatalf("padded splice wrong: %v", got[prefixLen:prefixLen+3])
	}
}

func TestPaddedLayoutConsumePlan_gemma4(t *testing.T) {
	t.Parallel()
	msgs := []api.Message{{
		Role:           "user",
		PaddedInputIDs: []int{1, 2},
		Images:         []api.ImageData{{1}},
	}}
	plan := PaddedLayoutConsumePlanForChat("gemma4-small", []string{"gemma4"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeGemma4Img {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestGemma4UserContentSpans(t *testing.T) {
	t.Parallel()
	rendered := "<bos><|turn>user\nbody1<turn|>\n<|turn>assistant\nok<turn|>\n<|turn>user\nbody2<turn|>\n"
	spans := gemma4UserContentSpans(rendered)
	if len(spans) != 2 {
		t.Fatalf("spans=%d want 2", len(spans))
	}
	if rendered[spans[0].contentStart:spans[0].contentEnd] != "body1" {
		t.Fatalf("span0=%q", rendered[spans[0].contentStart:spans[0].contentEnd])
	}
	if rendered[spans[1].contentStart:spans[1].contentEnd] != "body2" {
		t.Fatalf("span1=%q", rendered[spans[1].contentStart:spans[1].contentEnd])
	}
}
