package llamarunner

import (
	"errors"
	"fmt"

	"github.com/ollama/ollama/llm"
)

// Padded consume modes for native VLM families (mirror ollama-engine / llm package).
const (
	PaddedLayoutConsumeGemma3ImgRunner      = "gemma3_img_runner_inject"
	PaddedLayoutConsumeMllamaImgRunner      = "mllama_img_runner_inject"
	PaddedLayoutConsumeLlama4ImgRunner      = "llama4_img_runner_inject"
	PaddedLayoutConsumeLfm2ImgRunner        = "lfm2_img_runner_inject"
	PaddedLayoutConsumeGlmocrImgRunner      = "glmocr_img_runner_inject"
	PaddedLayoutConsumeMistral3ImgRunner    = "mistral3_img_runner_inject"
	PaddedLayoutConsumeDeepseekOcrImgRunner = "deepseekocr_img_runner_inject"
)

const (
	gemma3ImageSlotToken       = 255999
	mllamaImageSlotToken       = 128256
	llama4ImageBoundaryToken   = 200080
	llama4ImageToken           = 200090
	llama4PatchToken           = 200092
	lfm2ImageTokenFallback     = 396
	mistral3ImgTokenFallback   = 10
	mistral3BreakTokenFallback = 12
	mistral3EndTokenFallback   = 13
	deepseekOcrImageFallback   = 128815
)

type imageChunkCursor struct {
	chunks [][]visionChunk
	idx    int
}

func newImageChunkCursor(chunks [][]visionChunk) imageChunkCursor {
	return imageChunkCursor{chunks: chunks}
}

func (c *imageChunkCursor) appendNext(inputs []input) []input {
	if c.idx >= len(c.chunks) {
		return inputs
	}
	for _, ch := range c.chunks[c.idx] {
		inputs = appendVisionChunk(inputs, ch)
	}
	c.idx++
	return inputs
}

func inputsFromImageSlotTokens(promptTokens []int, imageChunks [][]visionChunk, slot int) []input {
	if slot == 0 {
		var inputs []input
		for _, t := range promptTokens {
			inputs = append(inputs, input{token: t})
		}
		return inputs
	}
	cursor := newImageChunkCursor(imageChunks)
	var inputs []input
	for _, t := range promptTokens {
		if t == slot {
			inputs = cursor.appendNext(inputs)
			continue
		}
		inputs = append(inputs, input{token: t})
	}
	return inputs
}

func inputsFromLlama4ImageBlocks(promptTokens []int, imageChunks [][]visionChunk) []input {
	cursor := newImageChunkCursor(imageChunks)
	var inputs []input
	for i := 0; i < len(promptTokens); i++ {
		if isLlama4ImageBlockStart(promptTokens, i) {
			inputs = cursor.appendNext(inputs)
			if end := llama4ImageBlockEndIndex(promptTokens, i); end > i {
				i = end
			}
			continue
		}
		inputs = append(inputs, input{token: promptTokens[i]})
	}
	return inputs
}

func isLlama4ImageBlockStart(tokens []int, i int) bool {
	if i >= len(tokens) || tokens[i] != llama4ImageBoundaryToken {
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
	if start >= len(tokens) || tokens[start] != llama4ImageBoundaryToken {
		return start
	}
	for j := start + 1; j < len(tokens); j++ {
		if tokens[j] == llama4ImageBoundaryToken {
			return j
		}
	}
	return start
}

type lfm2VisionTokens struct {
	Image    int
	Start    int
	End      int
	UseBlock bool
}

func inputsFromLfm2PaddedTokens(promptTokens []int, imageChunks [][]visionChunk, slots lfm2VisionTokens) []input {
	cursor := newImageChunkCursor(imageChunks)
	var inputs []input
	for i := 0; i < len(promptTokens); i++ {
		if slots.UseBlock && isLfm2ImageBlockStart(promptTokens, i, slots) {
			inputs = cursor.appendNext(inputs)
			if end := lfm2ImageBlockEndIndex(promptTokens, i, slots.End); end > i {
				i = end
			}
			continue
		}
		if !slots.UseBlock && promptTokens[i] == slots.Image && isFirstImageTokenInRun(promptTokens, i, slots.Image) {
			inputs = cursor.appendNext(inputs)
			i = skipImageTokenRun(promptTokens, i, slots.Image)
			continue
		}
		inputs = append(inputs, input{token: promptTokens[i]})
	}
	return inputs
}

func isLfm2ImageBlockStart(tokens []int, i int, slots lfm2VisionTokens) bool {
	if !slots.UseBlock || i >= len(tokens) || tokens[i] != slots.Start {
		return false
	}
	return lfm2ImageBlockEndIndex(tokens, i, slots.End) > i
}

func lfm2ImageBlockEndIndex(tokens []int, start int, endToken int) int {
	if start >= len(tokens) || endToken == 0 {
		return start
	}
	for j := start + 1; j < len(tokens); j++ {
		if tokens[j] == endToken {
			return j
		}
	}
	return start
}

func isFirstImageTokenInRun(tokens []int, i int, imageToken int) bool {
	return i == 0 || tokens[i-1] != imageToken
}

func skipImageTokenRun(tokens []int, start int, imageToken int) int {
	for i := start; i < len(tokens); i++ {
		if tokens[i] != imageToken {
			return i - 1
		}
	}
	return len(tokens) - 1
}

type mistral3VisionTokens struct {
	Img   int
	Break int
	End   int
}

func inputsFromMistral3PaddedTokens(promptTokens []int, imageChunks [][]visionChunk, slots mistral3VisionTokens) []input {
	cursor := newImageChunkCursor(imageChunks)
	var inputs []input
	for i := 0; i < len(promptTokens); i++ {
		if isMistral3ImageInjectStart(promptTokens, i, slots) {
			inputs = cursor.appendNext(inputs)
			if end := mistral3ImageBlockEndIndex(promptTokens, i, slots); end > i {
				i = end
			}
			continue
		}
		inputs = append(inputs, input{token: promptTokens[i]})
	}
	return inputs
}

func isMistral3ImageInjectStart(tokens []int, i int, slots mistral3VisionTokens) bool {
	if i >= len(tokens) || tokens[i] != slots.Img {
		return false
	}
	if i > 0 && (tokens[i-1] == slots.Img || tokens[i-1] == slots.Break) {
		return false
	}
	return mistral3ImageBlockEndIndex(tokens, i, slots) > i
}

func mistral3ImageBlockEndIndex(tokens []int, start int, slots mistral3VisionTokens) int {
	for j := start; j < len(tokens); j++ {
		if tokens[j] == slots.End {
			return j
		}
	}
	return start
}

func inputsFromDeepseekOcrPaddedTokens(promptTokens []int, imageChunks [][]visionChunk, imageToken int) []input {
	cursor := newImageChunkCursor(imageChunks)
	var inputs []input
	for i := 0; i < len(promptTokens); i++ {
		if promptTokens[i] == imageToken && isFirstImageTokenInRun(promptTokens, i, imageToken) {
			inputs = cursor.appendNext(inputs)
			i = skipImageTokenRun(promptTokens, i, imageToken)
			continue
		}
		inputs = append(inputs, input{token: promptTokens[i]})
	}
	return inputs
}

func (s *Server) lfm2VisionTokens() (lfm2VisionTokens, error) {
	slots := lfm2VisionTokens{Image: lfm2ImageTokenFallback}
	if id, err := s.placeholderToken("<image>"); err == nil && id != 0 {
		slots.Image = id
	}
	start, err := s.placeholderToken("<|image_start|>")
	if err != nil {
		return lfm2VisionTokens{}, err
	}
	end, err := s.placeholderToken("<|image_end|>")
	if err != nil {
		return lfm2VisionTokens{}, err
	}
	slots.Start = start
	slots.End = end
	slots.UseBlock = start != 0 && end != 0
	return slots, nil
}

func (s *Server) mistral3VisionTokens() (mistral3VisionTokens, error) {
	slots := mistral3VisionTokens{
		Img:   mistral3ImgTokenFallback,
		Break: mistral3BreakTokenFallback,
		End:   mistral3EndTokenFallback,
	}
	if id, err := s.placeholderToken("[IMG]"); err == nil && id != 0 {
		slots.Img = id
	}
	if id, err := s.placeholderToken("[IMG_BREAK]"); err == nil && id != 0 {
		slots.Break = id
	}
	if id, err := s.placeholderToken("[IMG_END]"); err == nil && id != 0 {
		slots.End = id
	}
	return slots, nil
}

func (s *Server) deepseekOcrVisionToken() (int, error) {
	if id, err := s.placeholderToken("<image>"); err == nil && id != 0 {
		return id, nil
	}
	return deepseekOcrImageFallback, nil
}

func (s *Server) placeholderToken(placeholder string) (int, error) {
	toks, err := s.lc.Model().Tokenize(placeholder, false, true)
	if err != nil {
		return 0, err
	}
	if len(toks) == 0 {
		return 0, nil
	}
	return toks[len(toks)-1], nil
}

func supportsPaddedLayoutConsume(consume string) bool {
	switch consume {
	case PaddedLayoutConsumeQwen3VLHFRunner,
		PaddedLayoutConsumeGemma4ImgRunner,
		PaddedLayoutConsumeGemma3ImgRunner,
		PaddedLayoutConsumeMllamaImgRunner,
		PaddedLayoutConsumeLlama4ImgRunner,
		PaddedLayoutConsumeLfm2ImgRunner,
		PaddedLayoutConsumeGlmocrImgRunner,
		PaddedLayoutConsumeMistral3ImgRunner,
		PaddedLayoutConsumeDeepseekOcrImgRunner:
		return true
	default:
		return false
	}
}

func (s *Server) inputsFromPaddedLayoutConsume(
	promptTokens []int,
	images []llm.ImageData,
	consume string,
	sessionKey string,
	sessionOverlay bool,
	gemma4Media Gemma4PaddedMediaSchedule,
	deferEncode bool,
) ([]input, []deferredVisionImage, error) {
	if s.image == nil {
		return nil, nil, errors.New("padded prompt tokens require a vision model")
	}
	s.image.growCacheForDistinctFrames(len(images))
	imageChunks, deferred, err := s.imageChunksMaybeDeferred(images, sessionKey, sessionOverlay, deferEncode)
	if err != nil {
		return nil, nil, err
	}

	var inputs []input
	switch consume {
	case PaddedLayoutConsumeQwen3VLHFRunner:
		inputs = inputsFromQwen3VLPromptTokens(promptTokens, imageChunks)
	case PaddedLayoutConsumeGemma4ImgRunner:
		slots, err := s.gemma4SoftTokens()
		if err != nil {
			return nil, nil, err
		}
		schedule := Gemma4PaddedMediaSchedule{
			StillImageCount:  gemma4Media.StillImageCount,
			VideoFrameCounts: gemma4Media.VideoFrameCounts,
			AudioClipCount:   gemma4Media.AudioClipCount,
		}
		inputs = inputsFromGemma4PromptTokens(promptTokens, imageChunks, slots, schedule)
	case PaddedLayoutConsumeGemma3ImgRunner:
		slot := gemma3ImageSlotToken
		if id, err := s.placeholderToken("<start_of_image>"); err == nil && id != 0 {
			slot = id
		}
		inputs = inputsFromImageSlotTokens(promptTokens, imageChunks, slot)
	case PaddedLayoutConsumeMllamaImgRunner:
		slot := mllamaImageSlotToken
		if id, err := s.placeholderToken("<|image|>"); err == nil && id != 0 {
			slot = id
		}
		inputs = inputsFromImageSlotTokens(promptTokens, imageChunks, slot)
	case PaddedLayoutConsumeLlama4ImgRunner:
		inputs = inputsFromLlama4ImageBlocks(promptTokens, imageChunks)
	case PaddedLayoutConsumeLfm2ImgRunner, PaddedLayoutConsumeGlmocrImgRunner:
		slots, err := s.lfm2VisionTokens()
		if err != nil {
			return nil, nil, err
		}
		inputs = inputsFromLfm2PaddedTokens(promptTokens, imageChunks, slots)
	case PaddedLayoutConsumeMistral3ImgRunner:
		slots, err := s.mistral3VisionTokens()
		if err != nil {
			return nil, nil, err
		}
		inputs = inputsFromMistral3PaddedTokens(promptTokens, imageChunks, slots)
	case PaddedLayoutConsumeDeepseekOcrImgRunner:
		imageToken, err := s.deepseekOcrVisionToken()
		if err != nil {
			return nil, nil, err
		}
		inputs = inputsFromDeepseekOcrPaddedTokens(promptTokens, imageChunks, imageToken)
	default:
		return nil, nil, fmt.Errorf("unsupported padded layout consume mode: %s", consume)
	}

	if len(inputs) == 0 {
		return nil, nil, errors.New("no input provided")
	}
	return inputs, deferred, nil
}
