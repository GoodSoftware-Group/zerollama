package modality

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBuildDeepseekOcrPaddedCompletionPromptTokens_splice(t *testing.T) {
	msgs := []api.Message{{
		Role:           "user",
		Content:        "ocr this",
		PaddedInputIDs: []int{128815, 128815, 128815},
	}}
	rendered := "ocr this"
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeDeepseekOcrImg}
	got, ok := BuildDeepseekOcrPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	if got[0] != 128815 || got[2] != 128815 {
		t.Fatalf("padded splice wrong: %v", got[:3])
	}
}

func TestBuildDeepseekOcrPaddedCompletionPromptTokens_multiTurn(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "first", PaddedInputIDs: []int{1, 2}},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "second", PaddedInputIDs: []int{9, 8}},
	}
	rendered := "firstoksecond"
	plan := PaddedLayoutConsumePlan{Active: true, Mode: PaddedLayoutConsumeDeepseekOcrImg}
	got, ok := BuildDeepseekOcrPaddedCompletionPromptTokens(context.Background(), fakeTokenize, rendered, msgs, plan)
	if !ok {
		t.Fatal("expected ok")
	}
	if got[0] != 1 || got[1] != 2 {
		t.Fatalf("first user splice wrong: %v", got[:2])
	}
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

func TestPaddedLayoutConsumePlan_deepseekocr(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("", []string{"deepseekocr"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeDeepseekOcrImg {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestPaddedLayoutConsumePlan_qwen25vl_usesQwen3VLPath(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi", PaddedInputIDs: []int{1}, Images: []api.ImageData{{1}}}}
	plan := PaddedLayoutConsumePlanForChat("", []string{"qwen25vl"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeQwen3VLHF {
		t.Fatalf("qwen25vl should use Qwen3-VL consume path, plan=%+v", plan)
	}
}
