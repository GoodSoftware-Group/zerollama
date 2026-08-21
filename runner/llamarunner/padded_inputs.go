package llamarunner

// PaddedLayoutConsumeQwen3VLHFRunner is the consume mode for pretokenized SGLang layout inject.
const PaddedLayoutConsumeQwen3VLHFRunner = "qwen3vl_hf_runner_inject"

// PaddedLayoutConsumeGemma4ImgRunner injects ViT embeds at <|image|> soft tokens in pretokenized ids.
const PaddedLayoutConsumeGemma4ImgRunner = "gemma4_img_runner_inject"

const gemma4ImagePlaceholder = "<|image|>"
const gemma4VideoPlaceholder = "<|video|>"
const gemma4AudioPlaceholder = "<|audio|>"

const (
	qwenVLVisionStart = 151652
	qwenVLImagePad    = 151655
	qwenVLVisionEnd   = 151653
)

// visionChunk is a text or embed slice from mtmd for one image.
type visionChunk struct {
	tokens []int
	embed  []float32
	hash   uint64
}

func stampVisionChunkHash(vc []visionChunk, hash uint64) []visionChunk {
	for i := range vc {
		vc[i].hash = hash
	}
	return vc
}

func appendVisionChunk(inputs []input, c visionChunk) []input {
	if c.hash != 0 && len(c.embed) == 0 && len(c.tokens) == 0 {
		return append(inputs, input{embedHash: c.hash})
	}
	if len(c.embed) != 0 {
		return append(inputs, input{embed: c.embed, embedHash: c.hash})
	}
	for _, t := range c.tokens {
		inputs = append(inputs, input{token: t})
	}
	return inputs
}

// inputsFromQwen3VLPromptTokens maps pretokenized ids to runner inputs.
// When <|vision_start|> (151652) is present, each block consumes the next image's
// full mtmd chunk list and skips the padded tokens through <|vision_end|> (151653).
// Otherwise falls back to one embed per <|image_pad|> (151655).
func inputsFromQwen3VLPromptTokens(promptTokens []int, imageChunks [][]visionChunk) []input {
	if visionBlockInject(promptTokens) {
		return inputsFromQwen3VLVisionBlocks(promptTokens, imageChunks)
	}
	return inputsFromQwen3VLPadTokens(promptTokens, flatEmbeds(imageChunks))
}

func visionBlockInject(promptTokens []int) bool {
	for _, t := range promptTokens {
		if t == qwenVLVisionStart {
			return true
		}
	}
	return false
}

func inputsFromQwen3VLVisionBlocks(promptTokens []int, imageChunks [][]visionChunk) []input {
	var inputs []input
	imageIdx := 0
	for i := 0; i < len(promptTokens); i++ {
		t := promptTokens[i]
		if t == qwenVLVisionStart && imageIdx < len(imageChunks) {
			for _, c := range imageChunks[imageIdx] {
				inputs = appendVisionChunk(inputs, c)
			}
			imageIdx++
			for i < len(promptTokens) && promptTokens[i] != qwenVLVisionEnd {
				i++
			}
			continue
		}
		inputs = append(inputs, input{token: t})
	}
	return inputs
}

func inputsFromQwen3VLPadTokens(promptTokens []int, embedQueue [][]float32) []input {
	var inputs []input
	embedIdx := 0
	for _, t := range promptTokens {
		if t == qwenVLImagePad && embedIdx < len(embedQueue) {
			inputs = append(inputs, input{embed: embedQueue[embedIdx]})
			embedIdx++
			continue
		}
		inputs = append(inputs, input{token: t})
	}
	return inputs
}

func flatEmbeds(imageChunks [][]visionChunk) [][]float32 {
	var out [][]float32
	for _, chunks := range imageChunks {
		for _, c := range chunks {
			if len(c.embed) != 0 {
				out = append(out, c.embed)
			}
		}
	}
	return out
}

// Gemma4SoftTokens holds runtime-resolved Gemma4 multimodal soft token ids.
type Gemma4SoftTokens struct {
	Image int
	Video int
	Audio int
}

// Gemma4PaddedMediaSchedule orders flat imageChunks for padded inject (see server/modality).
type Gemma4PaddedMediaSchedule struct {
	StillImageCount  int
	VideoFrameCounts []int
	AudioClipCount   int
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

// inputsFromGemma4PromptTokens maps pretokenized ids to runner inputs, injecting mtmd
// chunks at <|image|>, <|video|>, and <|audio|> soft tokens (SGLang padded_input_ids).
func inputsFromGemma4PromptTokens(
	promptTokens []int,
	imageChunks [][]visionChunk,
	slots Gemma4SoftTokens,
	schedule Gemma4PaddedMediaSchedule,
) []input {
	if slots.Image == 0 && slots.Video == 0 && slots.Audio == 0 {
		var inputs []input
		for _, t := range promptTokens {
			inputs = append(inputs, input{token: t})
		}
		return inputs
	}
	if slots.Video == 0 && slots.Audio == 0 {
		return inputsFromSoftTokenSlots(promptTokens, flatEmbeds(imageChunks), slots.Image)
	}
	return inputsFromGemma4MultimodalSlots(promptTokens, imageChunks, slots, schedule)
}

type gemma4MediaCursor struct {
	chunks   [][]visionChunk
	idx      int
	videoIdx int
}

func newGemma4MediaCursor(chunks [][]visionChunk) gemma4MediaCursor {
	return gemma4MediaCursor{chunks: chunks}
}

func (c *gemma4MediaCursor) appendRaster(inputs []input) []input {
	if c.idx >= len(c.chunks) {
		return inputs
	}
	for _, ch := range c.chunks[c.idx] {
		inputs = appendVisionChunk(inputs, ch)
	}
	c.idx++
	return inputs
}

func (c *gemma4MediaCursor) appendVideoClip(inputs []input, frameCount int) []input {
	if frameCount <= 0 {
		frameCount = 1
	}
	for range frameCount {
		inputs = c.appendRaster(inputs)
	}
	c.videoIdx++
	return inputs
}

func inputsFromGemma4MultimodalSlots(
	promptTokens []int,
	imageChunks [][]visionChunk,
	slots Gemma4SoftTokens,
	schedule Gemma4PaddedMediaSchedule,
) []input {
	var inputs []input
	cursor := newGemma4MediaCursor(imageChunks)
	stillLeft := schedule.StillImageCount
	audioLeft := schedule.AudioClipCount

	for _, t := range promptTokens {
		switch t {
		case slots.Image:
			if stillLeft > 0 {
				stillLeft--
			}
			inputs = cursor.appendRaster(inputs)
		case slots.Video:
			frameCount := 1
			if cursor.videoIdx < len(schedule.VideoFrameCounts) {
				frameCount = schedule.VideoFrameCounts[cursor.videoIdx]
			}
			inputs = cursor.appendVideoClip(inputs, frameCount)
		case slots.Audio:
			if audioLeft > 0 {
				audioLeft--
			}
			inputs = cursor.appendRaster(inputs)
		default:
			inputs = append(inputs, input{token: t})
		}
	}
	return inputs
}

func inputsFromSoftTokenSlots(promptTokens []int, embedQueue [][]float32, slotToken int) []input {
	var inputs []input
	embedIdx := 0
	for _, t := range promptTokens {
		if t == slotToken && embedIdx < len(embedQueue) {
			inputs = append(inputs, input{embed: embedQueue[embedIdx]})
			embedIdx++
			continue
		}
		inputs = append(inputs, input{token: t})
	}
	return inputs
}
