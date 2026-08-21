package renderers

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/modality"
)

type Qwen3VLRenderer struct {
	isThinking bool

	emitEmptyThinkOnNoThink bool
	useImgTags              bool
}

func (r *Qwen3VLRenderer) LeadingBOS() string {
	return ""
}

func (r *Qwen3VLRenderer) renderContent(msgs []api.Message, msgIdx int, imageOffset int) (string, int) {
	content := msgs[msgIdx]
	// SGLang preprocessed path: padded_input_ids already includes vision placeholder
	// tokens. Skip duplicate blocks only when splice/inject will consume the layout.
	// WHY MessageSkipsVisionPlaceholdersForChat (not unconditional skip): tool-calling
	// agents can fail multi-turn splice; prior turns must keep [img-N] / vision blocks
	// when role=tool is present. See modality.MessageSkipsVisionPlaceholdersForChat.
	if modality.MessageSkipsVisionPlaceholdersForChat(msgs, msgIdx, r.useImgTags) {
		var subSb strings.Builder
		subSb.WriteString(content.Content)
		return subSb.String(), imageOffset
	}
	// This assumes all images are at the front of the message - same assumption as ollama/ollama/runner.go
	var subSb strings.Builder
	emitOne := func() {
		if r.useImgTags {
			subSb.WriteString(fmt.Sprintf("[img-%d]", imageOffset))
			imageOffset++
		} else {
			subSb.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
		}
	}
	// When VideoSpans is set, Images are [still images..., frames from clip 0..., clip 1...].
	// HF path (!useImgTags): one <|vision_start|><|image_pad|><|vision_end|> per frame — matches
	// SGLang Qwen-VL (per-frame vision blocks, not a single grouped <|video|> token).
	// Production (useImgTags): [img-N] per raster for the flat mtmd image list.
	if len(content.VideoSpans) > 0 {
		videoFrames := 0
		for _, sp := range content.VideoSpans {
			videoFrames += sp.FrameCount
		}
		still := len(content.Images) - videoFrames
		if still < 0 || still > len(content.Images) {
			// Malformed metadata: fall back to flat image list.
			for range content.Images {
				emitOne()
			}
		} else {
			for i := 0; i < still; i++ {
				emitOne()
			}
			// One vision block per expanded video frame; ordering matches Images after stills then each clip.
			for _, sp := range content.VideoSpans {
				for range sp.FrameCount {
					emitOne()
				}
			}
		}
	} else {
		for range content.Images {
			emitOne()
		}
	}
	// Video is expanded to frames upstream; placeholders match the flat image list (grouped when VideoSpans is set).

	subSb.WriteString(content.Content)
	return subSb.String(), imageOffset
}

func (r *Qwen3VLRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder

	isThinking := r.isThinking
	if think != nil {
		isThinking = think.Bool()
	}

	if len(tools) > 0 {
		sb.WriteString(imStartTag + "system\n")
		if len(messages) > 0 && messages[0].Role == "system" {
			sb.WriteString(messages[0].Content + "\n\n")
		}
		sb.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, tool := range tools {
			sb.WriteString("\n")
			if b, err := marshalWithSpaces(tool); err == nil {
				sb.Write(b)
			}
		}
		sb.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	} else if len(messages) > 0 && messages[0].Role == "system" {
		sb.WriteString("<|im_start|>system\n" + messages[0].Content + "<|im_end|>\n")
	}
	multiStepTool := true
	lastQueryIndex := len(messages) - 1 // so this is the last user message

	for i := len(messages) - 1; i >= 0; i-- {
		message := messages[i]
		if multiStepTool && message.Role == "user" {
			// Check if content starts with <tool_response> and ends with </tool_response>
			content, _ := r.renderContent(messages, i, 0)
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				multiStepTool = false
				lastQueryIndex = i
			}
		}
	}

	imageOffset := 0
	for i, message := range messages {
		content, nextImageOffset := r.renderContent(messages, i, imageOffset)
		imageOffset = nextImageOffset

		lastMessage := i == len(messages)-1
		prefill := lastMessage && message.Role == "assistant"

		if message.Role == "user" || message.Role == "system" && i != 0 {
			sb.WriteString("<|im_start|>" + message.Role + "\n" + content + "<|im_end|>\n")
		} else if message.Role == "assistant" {
			contentReasoning := ""

			if isThinking {
				if message.Thinking != "" {
					contentReasoning = message.Thinking
				}
			}

			if isThinking && (i > lastQueryIndex || (contentReasoning != "" && len(message.ToolCalls) == 0)) {
				if i == len(messages)-1 || contentReasoning != "" {
					sb.WriteString("<|im_start|>" + message.Role + "\n<think>\n" + strings.Trim(contentReasoning, "\n")) // do we want to add a new line here?
					if content != "" {
						sb.WriteString("\n</think>\n\n" + strings.TrimLeft(content, "\n"))
					}
				} else {
					sb.WriteString("<|im_start|>" + message.Role + "\n" + content)
				}
			} else {
				sb.WriteString("<|im_start|>" + message.Role + "\n" + content)
			}

			if len(message.ToolCalls) > 0 {
				for j, toolCall := range message.ToolCalls {
					if j > 0 || content != "" {
						sb.WriteString("\n")
					}

					sb.WriteString("<tool_call>\n{\"name\": \"" + toolCall.Function.Name + "\", \"arguments\": ")
					if b, err := marshalWithSpaces(toolCall.Function.Arguments); err == nil {
						sb.Write(b)
					}
					sb.WriteString("}\n</tool_call>")
				}
			}

			if !prefill {
				sb.WriteString("<|im_end|>\n")
			}
		} else if message.Role == "tool" {
			// SGLang #33898: use renderContent so tool-result images emit vision
			// placeholders (bare message.Content advanced imageOffset without tags).
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<|im_start|>user")
			}
			sb.WriteString("\n<tool_response>\n" + content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString("<|im_end|>\n")
			}
		}

		// prefill at the end
		if lastMessage && !prefill {
			sb.WriteString("<|im_start|>assistant\n")
			if isThinking {
				sb.WriteString("<think>\n")
			} else if r.emitEmptyThinkOnNoThink {
				sb.WriteString("<think>\n\n</think>\n\n")
			}
		}
	}

	return sb.String(), nil
}
