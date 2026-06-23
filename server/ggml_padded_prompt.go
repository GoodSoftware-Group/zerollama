package server

import (
	"context"
	"log/slog"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/server/modality"
)

type paddedPromptBuilder func(
	context.Context,
	func(context.Context, string) ([]int, error),
	string,
	[]api.Message,
	modality.PaddedLayoutConsumePlan,
) ([]int, bool)

// ggmlPaddedCompletionPromptTokens builds pretokenized prompt ids for ggml runner
// inject when Qwen3-VL, Gemma4, mllama, Gemma3, Llama4, LFM2, GLM-OCR, Mistral3, or DeepSeek-OCR padded layout is active.
//
// WHY downgrade to deferred_multimodal_history: splice failure after render must
// not report runner_inject while sending a string prompt missing both layout ids
// and vision placeholders — operators grep layout_consume and warn logs.
func ggmlPaddedCompletionPromptTokens(
	ctx context.Context,
	m *Model,
	tokenize tokenizeFunc,
	rendered string,
	msgs []api.Message,
	plan modality.PaddedLayoutConsumePlan,
) ([]int, modality.PaddedLayoutConsumeMode) {
	if m == nil || m.IsMLX() || !plan.Active {
		return nil, plan.Mode
	}
	switch plan.Mode {
	case modality.PaddedLayoutConsumeQwen3VLHF:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildPaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeQwen3VLHFRunner,
		)
	case modality.PaddedLayoutConsumeGemma4Img:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildGemma4PaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeGemma4ImgRunner,
		)
	case modality.PaddedLayoutConsumeMllamaImg:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildMllamaPaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeMllamaImgRunner,
		)
	case modality.PaddedLayoutConsumeGemma3Img:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildGemma3PaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeGemma3ImgRunner,
		)
	case modality.PaddedLayoutConsumeLlama4Img:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildLlama4PaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeLlama4ImgRunner,
		)
	case modality.PaddedLayoutConsumeLfm2Img:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildLfm2PaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeLfm2ImgRunner,
		)
	case modality.PaddedLayoutConsumeGlmocrImg:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildGlmocrPaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeGlmocrImgRunner,
		)
	case modality.PaddedLayoutConsumeMistral3Img:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildMistral3PaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeMistral3ImgRunner,
		)
	case modality.PaddedLayoutConsumeDeepseekOcrImg:
		return tryGgmlPaddedInject(ctx, tokenize, rendered, msgs, plan,
			modality.BuildDeepseekOcrPaddedCompletionPromptTokens,
			modality.PaddedLayoutConsumeDeepseekOcrImgRunner,
		)
	default:
		return nil, plan.Mode
	}
}

func tryGgmlPaddedInject(
	ctx context.Context,
	tokenize tokenizeFunc,
	rendered string,
	msgs []api.Message,
	plan modality.PaddedLayoutConsumePlan,
	build paddedPromptBuilder,
	runnerMode modality.PaddedLayoutConsumeMode,
) ([]int, modality.PaddedLayoutConsumeMode) {
	ids, ok := build(ctx, tokenize, rendered, msgs, plan)
	if !ok || len(ids) == 0 {
		if plan.Active {
			slog.Warn("padded_input_ids runner inject unavailable; deferring multimodal history render",
				"prior_user_padded", modality.HasPriorUserPaddedInputIDs(msgs),
				"layout_consume", string(plan.Mode),
			)
			return nil, modality.PaddedLayoutConsumeDeferredHistory
		}
		return nil, plan.Mode
	}
	return ids, runnerMode
}

func llmGemma4PaddedMediaSchedule(paddedLayoutConsume string, msgs []api.Message) llm.Gemma4PaddedMediaSchedule {
	if paddedLayoutConsume != string(modality.PaddedLayoutConsumeGemma4ImgRunner) {
		return llm.Gemma4PaddedMediaSchedule{}
	}
	s := modality.Gemma4PaddedMediaScheduleForChat(msgs)
	return llm.Gemma4PaddedMediaSchedule{
		StillImageCount:  s.StillImageCount,
		VideoFrameCounts: append([]int(nil), s.VideoFrameCounts...),
		AudioClipCount:   s.AudioClipCount,
	}
}
