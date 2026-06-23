package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPaddedLayoutConsumePlan_ollamaEngineVLFamilies(t *testing.T) {
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{1},
		Images:         []api.ImageData{{1}},
	}}
	cases := []struct {
		name      string
		renderer  string
		families  []string
		wantMode  PaddedLayoutConsumeMode
	}{
		{"qwen3vl", "qwen3-vl-instruct", []string{"qwen3vl"}, PaddedLayoutConsumeQwen3VLHF},
		{"qwen25vl", "", []string{"qwen25vl"}, PaddedLayoutConsumeQwen3VLHF},
		{"qwen2vl", "", []string{"qwen2vl"}, PaddedLayoutConsumeQwen3VLHF},
		{"gemma4", "gemma4", []string{"gemma4"}, PaddedLayoutConsumeGemma4Img},
		{"mllama", "", []string{"mllama"}, PaddedLayoutConsumeMllamaImg},
		{"gemma3", "gemma3", []string{"gemma3"}, PaddedLayoutConsumeGemma3Img},
		{"llama4", "", []string{"llama4"}, PaddedLayoutConsumeLlama4Img},
		{"lfm2", "lfm2", []string{"lfm2"}, PaddedLayoutConsumeLfm2Img},
		{"glmocr", "glm-ocr", []string{"glmocr"}, PaddedLayoutConsumeGlmocrImg},
		{"mistral3", "", []string{"mistral3"}, PaddedLayoutConsumeMistral3Img},
		{"deepseekocr", "", []string{"deepseekocr"}, PaddedLayoutConsumeDeepseekOcrImg},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan := PaddedLayoutConsumePlanForChat(tc.renderer, tc.families, msgs, true)
			if !plan.Active || plan.Mode != tc.wantMode {
				t.Fatalf("plan=%+v want %q", plan, tc.wantMode)
			}
		})
	}
}

func TestPaddedLayoutConsumePlan_stillDeferred(t *testing.T) {
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{1},
		Images:         []api.ImageData{{1}},
	}}
	plan := PaddedLayoutConsumePlanForChat("", []string{"gemma3n"}, msgs, true)
	if !plan.Active || plan.Mode != PaddedLayoutConsumeDeferredNonQwen3V {
		t.Fatalf("gemma3n (text-only) should defer, plan=%+v", plan)
	}
}
