package ollamarunner

import (
	"fmt"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/ml"
)

const (
	PaddedLayoutConsumeQwen3VLHFRunner  = "qwen3vl_hf_runner_inject"
	PaddedLayoutConsumeGemma4ImgRunner  = "gemma4_img_runner_inject"
	PaddedLayoutConsumeMllamaImgRunner  = "mllama_img_runner_inject"
	PaddedLayoutConsumeGemma3ImgRunner  = "gemma3_img_runner_inject"
	PaddedLayoutConsumeLlama4ImgRunner  = "llama4_img_runner_inject"
	PaddedLayoutConsumeLfm2ImgRunner      = "lfm2_img_runner_inject"
	PaddedLayoutConsumeGlmocrImgRunner    = "glmocr_img_runner_inject"
	PaddedLayoutConsumeMistral3ImgRunner  = "mistral3_img_runner_inject"
	PaddedLayoutConsumeDeepseekOcrImgRunner = "deepseekocr_img_runner_inject"
)

const (
	qwenVLVisionStart = 151652
	qwenVLVisionEnd   = 151653

	llama4ImageBoundary = 200080 // <|image_start|> and <|image_end|>
	llama4ImageToken    = 200090 // <|image|>
	llama4PatchToken    = 200092 // <|patch|>
)

// Gemma4PaddedMediaSchedule orders rasters for Gemma4 padded inject (mirrors llm package).
type Gemma4PaddedMediaSchedule struct {
	StillImageCount  int
	VideoFrameCounts []int
	AudioClipCount   int
}

type Gemma4SoftTokens struct {
	Image int
	Video int
	Audio int
}

// inputsFromPaddedPromptTokens builds engine inputs from pretokenized ids (SGLang padded_input_ids).
func (s *Server) inputsFromPaddedPromptTokens(
	promptTokens []int,
	images []llm.ImageData,
	consume string,
	gemma4Slots Gemma4SoftTokens,
	gemma4Media Gemma4PaddedMediaSchedule,
	lfm2Slots Lfm2VisionTokens,
	mistral3Slots Mistral3VisionTokens,
	deepseekOcrSlots DeepseekOcrVisionTokens,
	sessionKey string,
) ([]*input.Input, []ml.Context, multimodalStore, error) {
	mp, ok := s.model.(model.MultimodalProcessor)
	if !ok {
		return nil, nil, nil, fmt.Errorf("padded prompt tokens require a multimodal model")
	}

	var raw []*input.Input
	var ctxs []ml.Context
	mmStore := newMultimodalStore()
	imageIdx := 0

	appendImage := func() error {
		if imageIdx >= len(images) {
			return fmt.Errorf("padded inject needs image at index %d, got %d", imageIdx, len(images))
		}
		img := images[imageIdx]
		imageIdx++
		ctx := s.model.Backend().NewContext()
		mm, err := s.encodeMultimodalCached(ctx, img.Data, sessionKey)
		if err != nil {
			ctx.Close()
			return err
		}
		s.multimodalHash.Reset()
		_, _ = s.multimodalHash.Write(img.Data)
		raw = append(raw, &input.Input{Multimodal: mm, MultimodalHash: s.multimodalHash.Sum64()})
		mmStore.addMultimodal(mm)
		ctxs = append(ctxs, ctx)
		return nil
	}

	switch consume {
	case PaddedLayoutConsumeQwen3VLHFRunner:
		for i := 0; i < len(promptTokens); i++ {
			t := promptTokens[i]
			if t == qwenVLVisionStart {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				for i < len(promptTokens) && promptTokens[i] != qwenVLVisionEnd {
					i++
				}
				continue
			}
			raw = append(raw, &input.Input{Token: int32(t)})
		}
	case PaddedLayoutConsumeGemma4ImgRunner:
		videoIdx := 0
		for _, t := range promptTokens {
			switch t {
			case gemma4Slots.Image:
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
			case gemma4Slots.Video:
				n := 1
				if videoIdx < len(gemma4Media.VideoFrameCounts) {
					n = gemma4Media.VideoFrameCounts[videoIdx]
				}
				if n <= 0 {
					n = 1
				}
				for range n {
					if err := appendImage(); err != nil {
						return nil, nil, nil, err
					}
				}
				videoIdx++
			case gemma4Slots.Audio:
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
			default:
				raw = append(raw, &input.Input{Token: int32(t)})
			}
		}
	case PaddedLayoutConsumeMllamaImgRunner:
		slot := mllamaImageSlotToken()
		for _, t := range promptTokens {
			if slot != 0 && t == slot {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				continue
			}
			raw = append(raw, &input.Input{Token: int32(t)})
		}
	case PaddedLayoutConsumeGemma3ImgRunner:
		slot := gemma3ImageSlotToken()
		for _, t := range promptTokens {
			if slot != 0 && t == slot {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				continue
			}
			raw = append(raw, &input.Input{Token: int32(t)})
		}
	case PaddedLayoutConsumeLlama4ImgRunner:
		for i := 0; i < len(promptTokens); i++ {
			if isLlama4ImageBlockStart(promptTokens, i) {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				if end := llama4ImageBlockEndIndex(promptTokens, i); end > i {
					i = end
				}
				continue
			}
			raw = append(raw, &input.Input{Token: int32(promptTokens[i])})
		}
	case PaddedLayoutConsumeLfm2ImgRunner, PaddedLayoutConsumeGlmocrImgRunner:
		for i := 0; i < len(promptTokens); i++ {
			if lfm2Slots.UseBlock && isLfm2ImageBlockStart(promptTokens, i, lfm2Slots) {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				if end := lfm2ImageBlockEndIndex(promptTokens, i, lfm2Slots.End); end > i {
					i = end
				}
				continue
			}
			if !lfm2Slots.UseBlock && promptTokens[i] == lfm2Slots.Image && isFirstImageTokenInRun(promptTokens, i, lfm2Slots.Image) {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				i = skipImageTokenRun(promptTokens, i, lfm2Slots.Image)
				continue
			}
			raw = append(raw, &input.Input{Token: int32(promptTokens[i])})
		}
	case PaddedLayoutConsumeMistral3ImgRunner:
		for i := 0; i < len(promptTokens); i++ {
			if isMistral3ImageInjectStart(promptTokens, i, mistral3Slots) {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				if end := mistral3ImageBlockEndIndex(promptTokens, i, mistral3Slots); end > i {
					i = end
				}
				continue
			}
			raw = append(raw, &input.Input{Token: int32(promptTokens[i])})
		}
	case PaddedLayoutConsumeDeepseekOcrImgRunner:
		for i := 0; i < len(promptTokens); i++ {
			if promptTokens[i] == deepseekOcrSlots.Image && isFirstImageTokenInRun(promptTokens, i, deepseekOcrSlots.Image) {
				if err := appendImage(); err != nil {
					return nil, nil, nil, err
				}
				i = skipImageTokenRun(promptTokens, i, deepseekOcrSlots.Image)
				continue
			}
			raw = append(raw, &input.Input{Token: int32(promptTokens[i])})
		}
	default:
		return nil, nil, nil, fmt.Errorf("unsupported padded layout consume mode: %s", consume)
	}

	out, err := mp.PostTokenize(raw)
	if err != nil {
		return nil, nil, nil, err
	}
	logVisionGridHintsFromInputs(images, out)
	return out, ctxs, mmStore, nil
}

func mllamaImageSlotToken() int {
	// Matches mllama.Model.PostTokenize placeholder for <|image|>.
	return 128256
}

func gemma3ImageSlotToken() int {
	// Matches gemma3.Model.PostTokenize <start_of_image> token.
	return 255999
}

func isLlama4ImageBlockStart(tokens []int, i int) bool {
	if i >= len(tokens) || tokens[i] != llama4ImageBoundary {
		return false
	}
	end := llama4ImageBlockEndIndex(tokens, i)
	if end <= i {
		return false
	}
	for k := i + 1; k < end; k++ {
		if tokens[k] == llama4PatchToken || tokens[k] == llama4ImageToken {
			return true
		}
	}
	return false
}

func llama4ImageBlockEndIndex(tokens []int, start int) int {
	if start >= len(tokens) || tokens[start] != llama4ImageBoundary {
		return start
	}
	for j := start + 1; j < len(tokens); j++ {
		if tokens[j] == llama4ImageBoundary {
			return j
		}
	}
	return start
}
