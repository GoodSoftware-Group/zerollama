package modality

import (
	"slices"

	"github.com/ollama/ollama/api"
)

// PaddedLayoutConsumeMode reports how far native path consumes pretokenized ids.
type PaddedLayoutConsumeMode string

const (
	PaddedLayoutConsumeNone              PaddedLayoutConsumeMode = ""
	PaddedLayoutConsumeDeferred          PaddedLayoutConsumeMode = "deferred"
	PaddedLayoutConsumeQwen3VLHF         PaddedLayoutConsumeMode = "qwen3vl_hf_skip_placeholders"
	PaddedLayoutConsumeQwen3VLHFRunner   PaddedLayoutConsumeMode = "qwen3vl_hf_runner_inject"
	PaddedLayoutConsumeGemma4Img         PaddedLayoutConsumeMode = "gemma4_img_skip_placeholders"
	PaddedLayoutConsumeGemma4ImgRunner   PaddedLayoutConsumeMode = "gemma4_img_runner_inject"
	PaddedLayoutConsumeMllamaImg         PaddedLayoutConsumeMode = "mllama_img_skip_placeholders"
	PaddedLayoutConsumeMllamaImgRunner   PaddedLayoutConsumeMode = "mllama_img_runner_inject"
	PaddedLayoutConsumeGemma3Img         PaddedLayoutConsumeMode = "gemma3_img_skip_placeholders"
	PaddedLayoutConsumeGemma3ImgRunner   PaddedLayoutConsumeMode = "gemma3_img_runner_inject"
	PaddedLayoutConsumeLlama4Img         PaddedLayoutConsumeMode = "llama4_img_skip_placeholders"
	PaddedLayoutConsumeLlama4ImgRunner   PaddedLayoutConsumeMode = "llama4_img_runner_inject"
	PaddedLayoutConsumeLfm2Img           PaddedLayoutConsumeMode = "lfm2_img_skip_placeholders"
	PaddedLayoutConsumeLfm2ImgRunner     PaddedLayoutConsumeMode = "lfm2_img_runner_inject"
	PaddedLayoutConsumeGlmocrImg         PaddedLayoutConsumeMode = "glmocr_img_skip_placeholders"
	PaddedLayoutConsumeGlmocrImgRunner   PaddedLayoutConsumeMode = "glmocr_img_runner_inject"
	PaddedLayoutConsumeMistral3Img       PaddedLayoutConsumeMode = "mistral3_img_skip_placeholders"
	PaddedLayoutConsumeMistral3ImgRunner PaddedLayoutConsumeMode = "mistral3_img_runner_inject"
	PaddedLayoutConsumeDeepseekOcrImg       PaddedLayoutConsumeMode = "deepseekocr_img_skip_placeholders"
	PaddedLayoutConsumeDeepseekOcrImgRunner PaddedLayoutConsumeMode = "deepseekocr_img_runner_inject"
	PaddedLayoutConsumeDeferredHistory   PaddedLayoutConsumeMode = "deferred_multimodal_history"
	PaddedLayoutConsumeDeferredNonQwen3V PaddedLayoutConsumeMode = "deferred_non_qwen3vl"
)

// PaddedLayoutConsumePlan is the runner/renderer decision for latest-user padded_input_ids.
type PaddedLayoutConsumePlan struct {
	Mode   PaddedLayoutConsumeMode
	Stub   PaddedLayoutRunnerStub
	Active bool
}

// PaddedLayoutConsumePlanForChat returns consume mode for a chat turn after expand.
func PaddedLayoutConsumePlanForChat(rendererName string, modelFamilies []string, msgs []api.Message, productionImgTags bool) PaddedLayoutConsumePlan {
	stub, ok := LatestUserPaddedLayout(&api.ChatRequest{Messages: msgs})
	if !ok {
		return PaddedLayoutConsumePlan{}
	}
	plan := PaddedLayoutConsumePlan{Stub: stub, Active: true}
	if isQwen3VLModel(rendererName, modelFamilies) {
		plan.Mode = PaddedLayoutConsumeQwen3VLHF
		return plan
	}
	if isGemma4Model(rendererName, modelFamilies) {
		plan.Mode = PaddedLayoutConsumeGemma4Img
		return plan
	}
	if isMllamaModel(modelFamilies) {
		plan.Mode = PaddedLayoutConsumeMllamaImg
		return plan
	}
	if isGemma3Model(rendererName, modelFamilies) {
		plan.Mode = PaddedLayoutConsumeGemma3Img
		return plan
	}
	if isLlama4Model(modelFamilies) {
		plan.Mode = PaddedLayoutConsumeLlama4Img
		return plan
	}
	if isLfm2Model(rendererName, modelFamilies) {
		plan.Mode = PaddedLayoutConsumeLfm2Img
		return plan
	}
	if isGlmocrModel(rendererName, modelFamilies) {
		plan.Mode = PaddedLayoutConsumeGlmocrImg
		return plan
	}
	if isMistral3Model(modelFamilies) {
		plan.Mode = PaddedLayoutConsumeMistral3Img
		return plan
	}
	if isDeepseekOcrModel(modelFamilies) {
		plan.Mode = PaddedLayoutConsumeDeepseekOcrImg
		return plan
	}
	plan.Mode = PaddedLayoutConsumeDeferredNonQwen3V
	return plan
}

func isGemma4Model(rendererName string, modelFamilies []string) bool {
	switch rendererName {
	case "gemma4", "gemma4-small", "gemma4-large":
		return true
	}
	return slices.Contains(modelFamilies, "gemma4")
}

func isMllamaModel(modelFamilies []string) bool {
	return slices.Contains(modelFamilies, "mllama")
}

func isGemma3Model(rendererName string, modelFamilies []string) bool {
	if rendererName == "gemma3" {
		return true
	}
	return slices.Contains(modelFamilies, "gemma3")
}

func isLlama4Model(modelFamilies []string) bool {
	return slices.Contains(modelFamilies, "llama4")
}

func isLfm2Model(rendererName string, modelFamilies []string) bool {
	switch rendererName {
	case "lfm2", "lfm2-thinking":
		return true
	}
	return slices.Contains(modelFamilies, "lfm2") || slices.Contains(modelFamilies, "lfm2moe")
}

func isGlmocrModel(rendererName string, modelFamilies []string) bool {
	if rendererName == "glm-ocr" {
		return true
	}
	return slices.Contains(modelFamilies, "glmocr")
}

func isMistral3Model(modelFamilies []string) bool {
	return slices.Contains(modelFamilies, "mistral3")
}

func isDeepseekOcrModel(modelFamilies []string) bool {
	return slices.Contains(modelFamilies, "deepseekocr")
}

func isQwen3VLModel(rendererName string, modelFamilies []string) bool {
	switch rendererName {
	case "qwen3-vl-instruct", "qwen3-vl-thinking":
		return true
	}
	for _, f := range modelFamilies {
		if f == "qwen3vl" || f == "qwen3vlmoe" {
			return true
		}
	}
	return slices.Contains(modelFamilies, "qwen2vl") || slices.Contains(modelFamilies, "qwen25vl")
}

// MessageSkipsVisionPlaceholders reports whether Qwen3-VL renderContent should omit
// vision blocks or [img-N] tags because padded_input_ids already encodes the layout.
func MessageSkipsVisionPlaceholders(msg api.Message, productionImgTags bool) bool {
	_ = productionImgTags
	if len(msg.PaddedInputIDs) == 0 {
		return false
	}
	return len(msg.Images) > 0 || len(msg.VideoSpans) > 0
}

// MessageSkipsVisionPlaceholdersForChat is the production gate for renderContent.
//
// WHY not skip on every padded message: if multi-turn splice fails after we
// stripped placeholders on prior user turns, the prompt has neither pretokenized
// ids nor [img-N] / vision blocks — silent broken VLMs. Latest user always skips
// when padded+media; prior padded users keep placeholders when role=tool exists so
// deferred_multimodal_history can still render history markers.
func MessageSkipsVisionPlaceholdersForChat(msgs []api.Message, msgIdx int, productionImgTags bool) bool {
	if msgIdx < 0 || msgIdx >= len(msgs) {
		return false
	}
	msg := msgs[msgIdx]
	if !MessageSkipsVisionPlaceholders(msg, productionImgTags) {
		return false
	}
	if msg.Role != "user" {
		return false
	}
	if msgIdx == lastUserMessageIndex(msgs) {
		return true
	}
	return !chatHasToolRole(msgs)
}

func chatHasToolRole(msgs []api.Message) bool {
	for _, m := range msgs {
		if m.Role == "tool" {
			return true
		}
	}
	return false
}
