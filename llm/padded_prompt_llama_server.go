package llm

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
)

// PaddedLayoutConsumeQwen3VLHFRunner matches llamarunner padded inject consume mode.
const PaddedLayoutConsumeQwen3VLHFRunner = "qwen3vl_hf_runner_inject"

// PaddedLayoutConsumeGemma4ImgRunner matches llamarunner Gemma4 soft-token inject.
const PaddedLayoutConsumeGemma4ImgRunner = "gemma4_img_runner_inject"

// PaddedLayoutConsumeMllamaImgRunner matches ollama-engine mllama soft-token inject.
const PaddedLayoutConsumeMllamaImgRunner = "mllama_img_runner_inject"

// PaddedLayoutConsumeGemma3ImgRunner matches ollama-engine gemma3 <start_of_image> inject.
const PaddedLayoutConsumeGemma3ImgRunner = "gemma3_img_runner_inject"

// PaddedLayoutConsumeLlama4ImgRunner matches ollama-engine Llama4 image_start…image_end inject.
const PaddedLayoutConsumeLlama4ImgRunner = "llama4_img_runner_inject"

// PaddedLayoutConsumeLfm2ImgRunner matches ollama-engine LFM2 padded inject.
const PaddedLayoutConsumeLfm2ImgRunner = "lfm2_img_runner_inject"

// PaddedLayoutConsumeGlmocrImgRunner matches ollama-engine GLM-OCR padded inject.
const PaddedLayoutConsumeGlmocrImgRunner = "glmocr_img_runner_inject"

// PaddedLayoutConsumeMistral3ImgRunner matches ollama-engine Mistral3/Pixtral padded inject.
const PaddedLayoutConsumeMistral3ImgRunner = "mistral3_img_runner_inject"

// PaddedLayoutConsumeDeepseekOcrImgRunner matches ollama-engine DeepSeek-OCR padded inject.
const PaddedLayoutConsumeDeepseekOcrImgRunner = "deepseekocr_img_runner_inject"

const (
	qwenVLVisionStart = 151652
	qwenVLVisionEnd   = 151653

	llama4ImageBoundary = 200080
	llama4ImageToken    = 200090
	llama4PatchToken    = 200092

	mllamaImageSlotTokenDefault = 128256
	gemma3ImageSlotTokenDefault = 255999
)

const gemma4ImagePlaceholder = "<|image|>"
const gemma4VideoPlaceholder = "<|video|>"
const gemma4AudioPlaceholder = "<|audio|>"

// Gemma4SoftTokens holds resolved soft token ids for padded multimodal inject.
type Gemma4SoftTokens struct {
	Image int
	Video int
	Audio int
}

// Lfm2VisionTokens holds runtime-resolved LFM2/GLM-OCR vision token ids.
type Lfm2VisionTokens struct {
	Image    int
	Start    int
	End      int
	UseBlock bool
}

// Mistral3VisionTokens holds Pixtral/Mistral3 vision token ids for padded inject.
type Mistral3VisionTokens struct {
	Img   int
	Break int
	End   int
}

// DeepseekOcrVisionTokens holds DeepSeek-OCR image placeholder token id.
type DeepseekOcrVisionTokens struct {
	Image int
}

// llamaServerPaddedInjectPrompt carries a multimodal prompt_string built from
// pretokenized Qwen3-VL layout; Completion attaches multimodal_data in marker order.
type llamaServerPaddedInjectPrompt struct {
	PromptString string
	MediaCount   int
}

type detokenizeFunc func(context.Context, []int) (string, error)

// completionMediaFromRequest returns Media when set, otherwise builds it from Images.
func completionMediaFromRequest(req CompletionRequest) []MediaData {
	if len(req.Media) > 0 {
		return req.Media
	}
	if len(req.Images) == 0 {
		return nil
	}
	media := make([]MediaData, len(req.Images))
	for i, img := range req.Images {
		media[i] = NewMediaData(img.ID, img.Data)
	}
	return media
}

func qwen3VLPromptHasVisionBlocks(tokens []int) bool {
	for _, t := range tokens {
		if t == qwenVLVisionStart {
			return true
		}
	}
	return false
}

func truncateCompletionTokens(tokens []int, limit, nKeep int) (out []int, truncated bool, original int) {
	// WHY vision-aware cut: pretokenized Qwen3-VL layouts embed vision blocks as
	// token ids; splitting through vision_start…vision_end orphans media markers
	// on the llama-server inject path. expandCutToWholeVisionBlocks widens the
	// discard window to drop or keep whole blocks.
	if len(tokens) <= limit {
		return tokens, false, 0
	}
	if nKeep < 0 {
		nKeep = len(tokens)
	}
	nKeep = min(nKeep, limit)
	discard := len(tokens) - limit
	cutStart := nKeep
	cutEnd := nKeep + discard
	cutStart, cutEnd = expandCutToWholeVisionBlocks(tokens, cutStart, cutEnd)
	out = make([]int, 0, limit)
	out = append(out, tokens[:cutStart]...)
	out = append(out, tokens[cutEnd:]...)
	if len(out) > limit {
		// Vision blocks expanded the discard window; trim tail to fit.
		out = out[:limit]
	}
	return out, true, len(tokens)
}

func qwen3VLVisionBlockRanges(tokens []int) [][2]int {
	var ranges [][2]int
	for i := 0; i < len(tokens); i++ {
		if tokens[i] != qwenVLVisionStart {
			continue
		}
		start := i
		for i < len(tokens) && tokens[i] != qwenVLVisionEnd {
			i++
		}
		if i < len(tokens) && tokens[i] == qwenVLVisionEnd {
			ranges = append(ranges, [2]int{start, i})
		}
	}
	return ranges
}

func expandCutToWholeVisionBlocks(tokens []int, cutStart, cutEnd int) (int, int) {
	for _, blk := range qwen3VLVisionBlockRanges(tokens) {
		blkStart, blkEnd := blk[0], blk[1]
		if cutStart <= blkEnd && cutEnd > blkStart {
			cutStart = min(cutStart, blkStart)
			cutEnd = max(cutEnd, blkEnd+1)
		}
	}
	return cutStart, cutEnd
}

// buildLlamaServerPaddedMultimodalPrompt maps pretokenized Qwen3-VL ids to a
// prompt_string with one llama-server media marker per <|vision_start|> block.
// Text spans between vision blocks are detokenized; padded tokens inside blocks are skipped.
//
// WHY require matching vision_end: emitting a marker without a complete block
// attached the wrong raster or left llama-server with dangling multimodal_data slots.
func buildLlamaServerPaddedMultimodalPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	mediaMarker string,
) (string, int, error) {
	if detokenize == nil {
		return "", 0, fmt.Errorf("detokenize is required for padded multimodal prompt")
	}
	if !qwen3VLPromptHasVisionBlocks(promptTokens) {
		return "", 0, fmt.Errorf("padded prompt tokens missing vision_start")
	}

	var b strings.Builder
	mediaCount := 0
	flushText := func(run []int) error {
		if len(run) == 0 {
			return nil
		}
		text, err := detokenize(ctx, run)
		if err != nil {
			return err
		}
		b.WriteString(text)
		return nil
	}

	var textRun []int
	for i := 0; i < len(promptTokens); i++ {
		t := promptTokens[i]
		if t == qwenVLVisionStart {
			if err := flushText(textRun); err != nil {
				return "", 0, err
			}
			textRun = nil
			j := i + 1
			for j < len(promptTokens) && promptTokens[j] != qwenVLVisionEnd {
				j++
			}
			if j < len(promptTokens) && promptTokens[j] == qwenVLVisionEnd {
				b.WriteString(mediaMarker)
				mediaCount++
				i = j
			} else {
				textRun = append(textRun, t)
			}
			continue
		}
		textRun = append(textRun, t)
	}
	if err := flushText(textRun); err != nil {
		return "", 0, err
	}
	if mediaCount == 0 {
		return "", 0, fmt.Errorf("padded prompt tokens produced no vision blocks")
	}
	return b.String(), mediaCount, nil
}

func gemma4PromptHasSoftSlots(tokens []int, slots Gemma4SoftTokens) bool {
	for _, t := range tokens {
		if slots.Image != 0 && t == slots.Image {
			return true
		}
		if slots.Video != 0 && t == slots.Video {
			return true
		}
		if slots.Audio != 0 && t == slots.Audio {
			return true
		}
	}
	return false
}

func gemma4PromptHasImageSlots(tokens []int, imageSlot int) bool {
	return gemma4PromptHasSoftSlots(tokens, Gemma4SoftTokens{Image: imageSlot})
}

// buildLlamaServerGemma4PaddedMultimodalPrompt maps pretokenized Gemma4 ids to a
// prompt_string with llama-server media markers at each multimodal soft token.
// <|video|> expands to one marker per frame in the matching VideoSpans clip.
func buildLlamaServerGemma4PaddedMultimodalPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	slots Gemma4SoftTokens,
	schedule Gemma4PaddedMediaSchedule,
	mediaMarker string,
) (string, int, error) {
	if detokenize == nil {
		return "", 0, fmt.Errorf("detokenize is required for padded multimodal prompt")
	}
	if !gemma4PromptHasSoftSlots(promptTokens, slots) {
		return "", 0, fmt.Errorf("padded prompt tokens missing gemma4 multimodal soft tokens")
	}

	var b strings.Builder
	mediaCount := 0
	videoIdx := 0
	flushText := func(run []int) error {
		if len(run) == 0 {
			return nil
		}
		text, err := detokenize(ctx, run)
		if err != nil {
			return err
		}
		b.WriteString(text)
		return nil
	}
	writeMarkers := func(n int) {
		for range n {
			b.WriteString(mediaMarker)
			mediaCount++
		}
	}

	var textRun []int
	for _, t := range promptTokens {
		switch t {
		case slots.Image, slots.Audio:
			if err := flushText(textRun); err != nil {
				return "", 0, err
			}
			textRun = nil
			writeMarkers(1)
		case slots.Video:
			if err := flushText(textRun); err != nil {
				return "", 0, err
			}
			textRun = nil
			frameCount := 1
			if videoIdx < len(schedule.VideoFrameCounts) {
				frameCount = schedule.VideoFrameCounts[videoIdx]
			}
			if frameCount <= 0 {
				frameCount = 1
			}
			writeMarkers(frameCount)
			videoIdx++
		default:
			textRun = append(textRun, t)
		}
	}
	if err := flushText(textRun); err != nil {
		return "", 0, err
	}
	if mediaCount == 0 {
		return "", 0, fmt.Errorf("padded prompt tokens produced no multimodal slots")
	}
	return b.String(), mediaCount, nil
}

func paddedInjectMediaPayloads(media []MediaData, mediaCount int) ([]string, error) {
	if mediaCount <= 0 {
		return nil, fmt.Errorf("padded inject requires at least one vision block")
	}
	if len(media) < mediaCount {
		return nil, fmt.Errorf("padded inject needs %d media payloads, got %d", mediaCount, len(media))
	}
	out := make([]string, mediaCount)
	for i := 0; i < mediaCount; i++ {
		data, err := llamaServerMediaBytes(media[i].Data)
		if err != nil {
			return nil, err
		}
		out[i] = base64.StdEncoding.EncodeToString(data)
	}
	return out, nil
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

func isLfm2ImageBlockStart(tokens []int, i int, slots Lfm2VisionTokens) bool {
	if !slots.UseBlock || i >= len(tokens) || tokens[i] != slots.Start {
		return false
	}
	return lfm2ImageBlockEndIndex(tokens, i, slots.End) > i
}

func lfm2PromptHasVisionSlots(tokens []int, slots Lfm2VisionTokens) bool {
	if slots.UseBlock {
		for i := range tokens {
			if isLfm2ImageBlockStart(tokens, i, slots) {
				return true
			}
		}
		return false
	}
	for i, t := range tokens {
		if t == slots.Image && isFirstImageTokenInRun(tokens, i, slots.Image) {
			return true
		}
	}
	return false
}

func mistral3ImageBlockEndIndex(tokens []int, start int, slots Mistral3VisionTokens) int {
	for j := start; j < len(tokens); j++ {
		if tokens[j] == slots.End {
			return j
		}
	}
	return start
}

func isMistral3ImageInjectStart(tokens []int, i int, slots Mistral3VisionTokens) bool {
	if i >= len(tokens) || tokens[i] != slots.Img {
		return false
	}
	if i > 0 && (tokens[i-1] == slots.Img || tokens[i-1] == slots.Break) {
		return false
	}
	return mistral3ImageBlockEndIndex(tokens, i, slots) > i
}

func mistral3PromptHasVisionBlocks(tokens []int, slots Mistral3VisionTokens) bool {
	for i := range tokens {
		if isMistral3ImageInjectStart(tokens, i, slots) {
			return true
		}
	}
	return false
}

func deepseekOcrPromptHasImageRuns(tokens []int, imageToken int) bool {
	for i, t := range tokens {
		if t == imageToken && isFirstImageTokenInRun(tokens, i, imageToken) {
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

func llama4PromptHasVisionBlocks(tokens []int) bool {
	for i := range tokens {
		if isLlama4ImageBlockStart(tokens, i) {
			return true
		}
	}
	return false
}

func promptHasSlotToken(tokens []int, slot int) bool {
	if slot == 0 {
		return false
	}
	for _, t := range tokens {
		if t == slot {
			return true
		}
	}
	return false
}

// buildLlamaServerBlockVisionPrompt maps start…end vision blocks to media markers.
func buildLlamaServerBlockVisionPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	startTok, endTok int,
	mediaMarker string,
	blockStart func([]int, int) bool,
	blockEnd func([]int, int) int,
) (string, int, error) {
	if detokenize == nil {
		return "", 0, fmt.Errorf("detokenize is required for padded multimodal prompt")
	}
	var b strings.Builder
	mediaCount := 0
	flushText := func(run []int) error {
		if len(run) == 0 {
			return nil
		}
		text, err := detokenize(ctx, run)
		if err != nil {
			return err
		}
		b.WriteString(text)
		return nil
	}

	var textRun []int
	for i := 0; i < len(promptTokens); i++ {
		if blockStart(promptTokens, i) {
			if err := flushText(textRun); err != nil {
				return "", 0, err
			}
			textRun = nil
			end := blockEnd(promptTokens, i)
			if end > i {
				b.WriteString(mediaMarker)
				mediaCount++
				i = end
			} else {
				textRun = append(textRun, promptTokens[i])
			}
			continue
		}
		textRun = append(textRun, promptTokens[i])
	}
	if err := flushText(textRun); err != nil {
		return "", 0, err
	}
	if mediaCount == 0 {
		return "", 0, fmt.Errorf("padded prompt tokens produced no vision blocks")
	}
	return b.String(), mediaCount, nil
}

func buildLlamaServerLfm2PaddedMultimodalPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	slots Lfm2VisionTokens,
	mediaMarker string,
) (string, int, error) {
	if slots.UseBlock {
		return buildLlamaServerBlockVisionPrompt(
			ctx, detokenize, promptTokens, slots.Start, slots.End, mediaMarker,
			func(tokens []int, i int) bool { return isLfm2ImageBlockStart(tokens, i, slots) },
			func(tokens []int, i int) int { return lfm2ImageBlockEndIndex(tokens, i, slots.End) },
		)
	}
	if detokenize == nil {
		return "", 0, fmt.Errorf("detokenize is required for padded multimodal prompt")
	}
	var b strings.Builder
	mediaCount := 0
	flushText := func(run []int) error {
		if len(run) == 0 {
			return nil
		}
		text, err := detokenize(ctx, run)
		if err != nil {
			return err
		}
		b.WriteString(text)
		return nil
	}
	var textRun []int
	for i := 0; i < len(promptTokens); i++ {
		if promptTokens[i] == slots.Image && isFirstImageTokenInRun(promptTokens, i, slots.Image) {
			if err := flushText(textRun); err != nil {
				return "", 0, err
			}
			textRun = nil
			b.WriteString(mediaMarker)
			mediaCount++
			i = skipImageTokenRun(promptTokens, i, slots.Image)
			continue
		}
		textRun = append(textRun, promptTokens[i])
	}
	if err := flushText(textRun); err != nil {
		return "", 0, err
	}
	if mediaCount == 0 {
		return "", 0, fmt.Errorf("padded prompt tokens produced no image runs")
	}
	return b.String(), mediaCount, nil
}

func buildLlamaServerMistral3PaddedMultimodalPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	slots Mistral3VisionTokens,
	mediaMarker string,
) (string, int, error) {
	return buildLlamaServerBlockVisionPrompt(
		ctx, detokenize, promptTokens, slots.Img, slots.End, mediaMarker,
		func(tokens []int, i int) bool { return isMistral3ImageInjectStart(tokens, i, slots) },
		func(tokens []int, i int) int { return mistral3ImageBlockEndIndex(tokens, i, slots) },
	)
}

func buildLlamaServerDeepseekOcrPaddedMultimodalPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	imageToken int,
	mediaMarker string,
) (string, int, error) {
	if detokenize == nil {
		return "", 0, fmt.Errorf("detokenize is required for padded multimodal prompt")
	}
	var b strings.Builder
	mediaCount := 0
	flushText := func(run []int) error {
		if len(run) == 0 {
			return nil
		}
		text, err := detokenize(ctx, run)
		if err != nil {
			return err
		}
		b.WriteString(text)
		return nil
	}
	var textRun []int
	for i := 0; i < len(promptTokens); i++ {
		if promptTokens[i] == imageToken && isFirstImageTokenInRun(promptTokens, i, imageToken) {
			if err := flushText(textRun); err != nil {
				return "", 0, err
			}
			textRun = nil
			b.WriteString(mediaMarker)
			mediaCount++
			i = skipImageTokenRun(promptTokens, i, imageToken)
			continue
		}
		textRun = append(textRun, promptTokens[i])
	}
	if err := flushText(textRun); err != nil {
		return "", 0, err
	}
	if mediaCount == 0 {
		return "", 0, fmt.Errorf("padded prompt tokens produced no image runs")
	}
	return b.String(), mediaCount, nil
}

func buildLlamaServerSlotPaddedMultimodalPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	slot int,
	mediaMarker string,
) (string, int, error) {
	if detokenize == nil {
		return "", 0, fmt.Errorf("detokenize is required for padded multimodal prompt")
	}
	if slot == 0 {
		return "", 0, fmt.Errorf("padded prompt tokens missing vision slot token")
	}
	var b strings.Builder
	mediaCount := 0
	flushText := func(run []int) error {
		if len(run) == 0 {
			return nil
		}
		text, err := detokenize(ctx, run)
		if err != nil {
			return err
		}
		b.WriteString(text)
		return nil
	}
	var textRun []int
	for _, t := range promptTokens {
		if t == slot {
			if err := flushText(textRun); err != nil {
				return "", 0, err
			}
			textRun = nil
			b.WriteString(mediaMarker)
			mediaCount++
			continue
		}
		textRun = append(textRun, t)
	}
	if err := flushText(textRun); err != nil {
		return "", 0, err
	}
	if mediaCount == 0 {
		return "", 0, fmt.Errorf("padded prompt tokens produced no vision slots")
	}
	return b.String(), mediaCount, nil
}

func buildLlamaServerLlama4PaddedMultimodalPrompt(
	ctx context.Context,
	detokenize detokenizeFunc,
	promptTokens []int,
	mediaMarker string,
) (string, int, error) {
	return buildLlamaServerBlockVisionPrompt(
		ctx, detokenize, promptTokens, llama4ImageBoundary, llama4ImageBoundary, mediaMarker,
		isLlama4ImageBlockStart,
		llama4ImageBlockEndIndex,
	)
}
