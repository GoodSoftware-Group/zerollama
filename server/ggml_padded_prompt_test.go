package server

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

func TestGgmlPaddedCompletionPromptTokens_qwen3vl(t *testing.T) {
	t.Parallel()
	rendered := "<|im_start|>user\n" + "hi" + "<|im_end|>\n<|im_start|>assistant\n"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{10, 11},
	}}
	plan := modality.PaddedLayoutConsumePlan{Active: true, Mode: modality.PaddedLayoutConsumeQwen3VLHF}
	m := &Model{ProjectorPaths: []string{"mmproj"}}
	ids, mode := ggmlPaddedCompletionPromptTokens(context.Background(), m, fakeServerTokenize, rendered, msgs, plan)
	if len(ids) == 0 || mode != modality.PaddedLayoutConsumeQwen3VLHFRunner {
		t.Fatalf("ids=%v mode=%q", ids, mode)
	}
}

func TestGgmlPaddedCompletionPromptTokens_gemma4(t *testing.T) {
	t.Parallel()
	rendered := "<bos><|turn>user\n" + "hi" + "<turn|>\n<|turn>model\n"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{10, 11},
	}}
	plan := modality.PaddedLayoutConsumePlan{Active: true, Mode: modality.PaddedLayoutConsumeGemma4Img}
	m := &Model{ProjectorPaths: []string{"mmproj"}}
	ids, mode := ggmlPaddedCompletionPromptTokens(context.Background(), m, fakeServerTokenize, rendered, msgs, plan)
	if len(ids) == 0 || mode != modality.PaddedLayoutConsumeGemma4ImgRunner {
		t.Fatalf("ids=%v mode=%q", ids, mode)
	}
}

func TestGgmlPaddedCompletionPromptTokens_lfm2(t *testing.T) {
	t.Parallel()
	rendered := "<|im_start|>user\n" + "hi" + "<|im_end|>\n<|im_start|>assistant\n"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{10, 11},
	}}
	plan := modality.PaddedLayoutConsumePlan{Active: true, Mode: modality.PaddedLayoutConsumeLfm2Img}
	m := &Model{ProjectorPaths: []string{"mmproj"}}
	ids, mode := ggmlPaddedCompletionPromptTokens(context.Background(), m, fakeServerTokenize, rendered, msgs, plan)
	if len(ids) == 0 || mode != modality.PaddedLayoutConsumeLfm2ImgRunner {
		t.Fatalf("ids=%v mode=%q", ids, mode)
	}
}

func TestGgmlPaddedCompletionPromptTokens_glmocr(t *testing.T) {
	t.Parallel()
	rendered := "[gMASK]<sop><|user|>\n" + "hi" + "<|assistant|>\n"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{10, 11},
	}}
	plan := modality.PaddedLayoutConsumePlan{Active: true, Mode: modality.PaddedLayoutConsumeGlmocrImg}
	m := &Model{ProjectorPaths: []string{"mmproj"}}
	ids, mode := ggmlPaddedCompletionPromptTokens(context.Background(), m, fakeServerTokenize, rendered, msgs, plan)
	if len(ids) == 0 || mode != modality.PaddedLayoutConsumeGlmocrImgRunner {
		t.Fatalf("ids=%v mode=%q", ids, mode)
	}
}

func TestGgmlPaddedCompletionPromptTokens_mistral3(t *testing.T) {
	t.Parallel()
	rendered := "<s>[INST] " + "hi" + " [/INST]"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{10, 13},
	}}
	plan := modality.PaddedLayoutConsumePlan{Active: true, Mode: modality.PaddedLayoutConsumeMistral3Img}
	m := &Model{ProjectorPaths: []string{"mmproj"}}
	ids, mode := ggmlPaddedCompletionPromptTokens(context.Background(), m, fakeServerTokenize, rendered, msgs, plan)
	if len(ids) == 0 || mode != modality.PaddedLayoutConsumeMistral3ImgRunner {
		t.Fatalf("ids=%v mode=%q", ids, mode)
	}
}

func TestGgmlPaddedCompletionPromptTokens_deepseekocr(t *testing.T) {
	t.Parallel()
	rendered := "hi"
	msgs := []api.Message{{
		Role:           "user",
		Content:        "hi",
		PaddedInputIDs: []int{128815, 128815},
	}}
	plan := modality.PaddedLayoutConsumePlan{Active: true, Mode: modality.PaddedLayoutConsumeDeepseekOcrImg}
	m := &Model{ProjectorPaths: []string{"mmproj"}}
	ids, mode := ggmlPaddedCompletionPromptTokens(context.Background(), m, fakeServerTokenize, rendered, msgs, plan)
	if len(ids) == 0 || mode != modality.PaddedLayoutConsumeDeepseekOcrImgRunner {
		t.Fatalf("ids=%v mode=%q", ids, mode)
	}
}

func TestGgmlPaddedCompletionPromptTokens_mlxSkipped(t *testing.T) {
	t.Parallel()
	m := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}}
	_, mode := ggmlPaddedCompletionPromptTokens(context.Background(), m, fakeServerTokenize, "", nil, modality.PaddedLayoutConsumePlan{
		Active: true,
		Mode:   modality.PaddedLayoutConsumeQwen3VLHF,
	})
	if mode != modality.PaddedLayoutConsumeQwen3VLHF {
		t.Fatalf("mlx should not upgrade mode, got %q", mode)
	}
}

func fakeServerTokenize(_ context.Context, s string) ([]int, error) {
	out := make([]int, len(s))
	for i := range s {
		out[i] = i + 1
	}
	return out, nil
}
