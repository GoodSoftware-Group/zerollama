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

const (
	qwenVLVisionStart = 151652
	qwenVLVisionEnd   = 151653
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
		out[i] = base64.StdEncoding.EncodeToString(media[i].Data)
	}
	return out, nil
}
