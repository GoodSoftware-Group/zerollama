package modality

import (
	"context"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
)

// Qwen3-VL chat template markers for splicing pretokenized user content.
const (
	qwenVLUserStart = "<|im_start|>user\n"
	qwenVLUserEnd   = "<|im_end|>"
	// Tool turns render as <|im_start|>user\n<tool_response>… — not a real user block for splice.
	qwenVLToolResponsePrefix = "<tool_response>"
)

// Gemma4 user-turn markers for pretokenized layout splice.
const (
	gemma4UserStart = "<|turn>user\n"
	gemma4UserEnd   = "<turn|>"
)

type userContentSpan struct {
	contentStart int
	contentEnd   int
}

// BuildPaddedCompletionPromptTokens splices padded_input_ids into each matching
// user block of a fully rendered Qwen3-VL chat prompt for ggml runner inject.
//
// WHY splice: padded_input_ids is the pretokenized multimodal slice per user
// turn; the rendered template still supplies im_start/im_end wrappers and
// assistant prefill prefix. Tokenizing inter-block text + padded slices avoids
// re-emitting vision placeholder text the renderer already skipped.
//
// WHY tool spans are excluded: Qwen3-VL renders tool results as
// <|im_start|>user\n<tool_response>… — they look like user blocks but are not
// role=user messages. Counting them breaks alignment with userMessageIndices and
// silently fails multi-turn agent history (renderer may have already skipped
// placeholders). See docs/sglang-multimodal-borrowings.md §25.
func BuildPaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeQwen3VLHF || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, qwenVLUserContentSpans(rendered), qwenVLUserContentBounds)
}

// BuildGemma4PaddedCompletionPromptTokens splices padded_input_ids into Gemma4 user turns.
//
// WHY Gemma4: SGLang preprocessed clients send per-frame soft tokens in padded_input_ids;
// production render uses [img-N] but inject replaces soft tokens at the runner (§30).
// Tool responses live inside assistant turns (<|tool_response>|), not <|turn>user — no
// pseudo-user span exclusion needed (unlike Qwen3-VL).
func BuildGemma4PaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeGemma4Img || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, gemma4UserContentSpans(rendered), gemma4UserContentBounds)
}

func buildPaddedCompletionPromptTokensForSpans(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	spans []userContentSpan,
	lastUserBounds func(string) (contentStart, contentEnd int, ok bool),
) ([]int, bool) {
	idx := lastUserMessageIndex(msgs)
	if idx < 0 || len(msgs[idx].PaddedInputIDs) == 0 {
		return nil, false
	}
	userIdxs := userMessageIndices(msgs)
	if len(spans) == 0 || len(userIdxs) != len(spans) {
		if priorUserPaddedInputIDs(msgs) {
			slog.Warn("padded_input_ids splice failed: user span count mismatch",
				"user_messages", len(userIdxs),
				"user_spans", len(spans),
				"prior_user_padded", true,
			)
			return nil, false
		}
		return buildPaddedCompletionPromptTokensLastUserWithBounds(ctx, tokenize, rendered, msgs, lastUserBounds)
	}
	var out []int
	pos := 0
	for i, span := range spans {
		prefix, err := tokenize(ctx, rendered[pos:span.contentStart])
		if err != nil {
			return nil, false
		}
		out = append(out, prefix...)
		msg := msgs[userIdxs[i]]
		if len(msg.PaddedInputIDs) > 0 {
			out = append(out, msg.PaddedInputIDs...)
		} else {
			mid, err := tokenize(ctx, rendered[span.contentStart:span.contentEnd])
			if err != nil {
				return nil, false
			}
			out = append(out, mid...)
		}
		pos = span.contentEnd
	}
	suffix, err := tokenize(ctx, rendered[pos:])
	if err != nil {
		return nil, false
	}
	out = append(out, suffix...)
	return out, true
}

func buildPaddedCompletionPromptTokensLastUserWithBounds(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	bounds func(string) (contentStart, contentEnd int, ok bool),
) ([]int, bool) {
	idx := lastUserMessageIndex(msgs)
	if idx < 0 {
		return nil, false
	}
	padded := msgs[idx].PaddedInputIDs
	if len(padded) == 0 {
		return nil, false
	}
	contentStart, contentEnd, ok := bounds(rendered)
	if !ok {
		return nil, false
	}
	prefix, err := tokenize(ctx, rendered[:contentStart])
	if err != nil {
		return nil, false
	}
	suffix, err := tokenize(ctx, rendered[contentEnd:])
	if err != nil {
		return nil, false
	}
	out := make([]int, 0, len(prefix)+len(padded)+len(suffix))
	out = append(out, prefix...)
	out = append(out, padded...)
	out = append(out, suffix...)
	return out, true
}

func qwenVLUserContentSpans(rendered string) []userContentSpan {
	// WHY skip <tool_response>: pseudo-user tool blocks must not participate in
	// padded_input_ids splice alignment (see BuildPaddedCompletionPromptTokens).
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], qwenVLUserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(qwenVLUserStart)
		endRel := strings.Index(rendered[contentStart:], qwenVLUserEnd)
		if endRel < 0 {
			break
		}
		contentEnd := contentStart + endRel
		if strings.HasPrefix(rendered[contentStart:contentEnd], qwenVLToolResponsePrefix) {
			searchFrom = contentEnd + len(qwenVLUserEnd)
			continue
		}
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd + len(qwenVLUserEnd)
	}
	return spans
}

func userMessageIndices(msgs []api.Message) []int {
	var idxs []int
	for i, msg := range msgs {
		if msg.Role == "user" {
			idxs = append(idxs, i)
		}
	}
	return idxs
}

// HasPriorUserPaddedInputIDs reports whether any user message before the latest carries padded_input_ids.
func HasPriorUserPaddedInputIDs(msgs []api.Message) bool {
	return priorUserPaddedInputIDs(msgs)
}

func priorUserPaddedInputIDs(msgs []api.Message) bool {
	last := lastUserMessageIndex(msgs)
	for i, msg := range msgs {
		if i == last {
			continue
		}
		if msg.Role == "user" && len(msg.PaddedInputIDs) > 0 {
			return true
		}
	}
	return false
}

func qwenVLUserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := qwenVLUserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}

func gemma4UserContentSpans(rendered string) []userContentSpan {
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], gemma4UserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(gemma4UserStart)
		endRel := strings.Index(rendered[contentStart:], gemma4UserEnd)
		if endRel < 0 {
			break
		}
		contentEnd := contentStart + endRel
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd + len(gemma4UserEnd)
	}
	return spans
}

// Gemma3 user-turn markers (template.Execute / gemma3-instruct).
const (
	gemma3UserStart = "<start_of_turn>user\n"
	gemma3UserEnd   = "<end_of_turn>"
)

// BuildGemma3PaddedCompletionPromptTokens splices padded_input_ids into Gemma3 user turns.
func BuildGemma3PaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeGemma3Img || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, gemma3UserContentSpans(rendered), gemma3UserContentBounds)
}

func gemma3UserContentSpans(rendered string) []userContentSpan {
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], gemma3UserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(gemma3UserStart)
		endRel := strings.Index(rendered[contentStart:], gemma3UserEnd)
		if endRel < 0 {
			break
		}
		contentEnd := contentStart + endRel
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd + len(gemma3UserEnd)
	}
	return spans
}

func gemma3UserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := gemma3UserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}

// Llama 4 chat template markers (llama.cpp LLM_CHAT_TEMPLATE_LLAMA4).
const (
	llama4UserStart = "<|header_start|>user<|header_end|>\n\n"
	llama4UserEnd   = "<|eot|>"
)

// BuildLlama4PaddedCompletionPromptTokens splices padded_input_ids into Llama4 user turns.
func BuildLlama4PaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeLlama4Img || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, llama4UserContentSpans(rendered), llama4UserContentBounds)
}

func llama4UserContentSpans(rendered string) []userContentSpan {
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], llama4UserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(llama4UserStart)
		endRel := strings.Index(rendered[contentStart:], llama4UserEnd)
		if endRel < 0 {
			break
		}
		contentEnd := contentStart + endRel
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd + len(llama4UserEnd)
	}
	return spans
}

func llama4UserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := llama4UserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}

// LFM2 chat template markers (ChatML-style; same user span delimiters as Qwen3-VL HF).
const (
	lfm2UserStart = "<|im_start|>user\n"
	lfm2UserEnd   = "<|im_end|>"
)

// BuildLfm2PaddedCompletionPromptTokens splices padded_input_ids into LFM2 user turns.
//
// WHY LFM2: OllamaEngineRequired VLMs send pretokenized image_start…image_end blocks (or
// contiguous <image> runs); runner inject replaces blocks with EncodeMultimodal (§33).
// Tool turns use role=tool with <|tool_response_start|> wrappers — not im_start|user spans.
func BuildLfm2PaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeLfm2Img || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, lfm2UserContentSpans(rendered), lfm2UserContentBounds)
}

func lfm2UserContentSpans(rendered string) []userContentSpan {
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], lfm2UserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(lfm2UserStart)
		endRel := strings.Index(rendered[contentStart:], lfm2UserEnd)
		if endRel < 0 {
			break
		}
		contentEnd := contentStart + endRel
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd + len(lfm2UserEnd)
	}
	return spans
}

func lfm2UserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := lfm2UserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}

// GLM-OCR user-turn markers ([gMASK]<sop> prefix; user blocks end at next role tag).
const glmocrUserStart = "<|user|>\n"

var glmocrUserEndMarkers = []string{"<|assistant|>", "<|observation|>", "<|system|>"}

// BuildGlmocrPaddedCompletionPromptTokens splices padded_input_ids into GLM-OCR user turns.
func BuildGlmocrPaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeGlmocrImg || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, glmocrUserContentSpans(rendered), glmocrUserContentBounds)
}

func glmocrUserContentSpans(rendered string) []userContentSpan {
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], glmocrUserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(glmocrUserStart)
		contentEnd := len(rendered)
		for _, marker := range glmocrUserEndMarkers {
			if endRel := strings.Index(rendered[contentStart:], marker); endRel >= 0 {
				candidate := contentStart + endRel
				if candidate < contentEnd {
					contentEnd = candidate
				}
			}
		}
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd
	}
	return spans
}

func glmocrUserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := glmocrUserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}

// Mistral3 / Pixtral user-turn markers (mistral-instruct jinja template).
const (
	mistral3UserStart = "[INST] "
	mistral3UserEnd   = " [/INST]"
)

// BuildMistral3PaddedCompletionPromptTokens splices padded_input_ids into Mistral3 user turns.
func BuildMistral3PaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeMistral3Img || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, mistral3UserContentSpans(rendered), mistral3UserContentBounds)
}

func mistral3UserContentSpans(rendered string) []userContentSpan {
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], mistral3UserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(mistral3UserStart)
		endRel := strings.Index(rendered[contentStart:], mistral3UserEnd)
		if endRel < 0 {
			break
		}
		contentEnd := contentStart + endRel
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd + len(mistral3UserEnd)
	}
	return spans
}

func mistral3UserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := mistral3UserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}

// BuildDeepseekOcrPaddedCompletionPromptTokens splices padded_input_ids into DeepSeek-OCR turns.
//
// WHY content-order spans: deepseek-ocr jinja has no role wrappers — rendered prompt is a
// concatenation of per-message content strings (see llama.cpp LLM_CHAT_TEMPLATE_DEEPSEEK_OCR).
func BuildDeepseekOcrPaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeDeepseekOcrImg || !plan.Active || tokenize == nil {
		return nil, false
	}
	bounds := func(r string) (contentStart, contentEnd int, ok bool) {
		spans := deepseekOcrUserContentSpans(r, msgs)
		if len(spans) == 0 {
			return 0, 0, false
		}
		last := spans[len(spans)-1]
		return last.contentStart, last.contentEnd, true
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, deepseekOcrUserContentSpans(rendered, msgs), bounds)
}

func deepseekOcrUserContentSpans(rendered string, msgs []api.Message) []userContentSpan {
	var spans []userContentSpan
	pos := 0
	for _, msg := range msgs {
		content := msg.Content
		if content == "" {
			continue
		}
		rel := strings.Index(rendered[pos:], content)
		if rel < 0 {
			if msg.Role == "user" && priorUserPaddedInputIDs(msgs) {
				return spans
			}
			break
		}
		start := pos + rel
		if msg.Role == "user" {
			spans = append(spans, userContentSpan{contentStart: start, contentEnd: start + len(content)})
		}
		pos = start + len(content)
	}
	return spans
}

// Llama 3 / mllama user-turn markers (template.Execute output).
const (
	mllamaUserStart = "<|start_header_id|>user<|end_header_id|>"
	mllamaUserEnd   = "<|eot_id|>"
)

// BuildMllamaPaddedCompletionPromptTokens splices padded_input_ids into Llama 3 user turns.
func BuildMllamaPaddedCompletionPromptTokens(
	ctx context.Context,
	tokenize func(context.Context, string) ([]int, error),
	rendered string,
	msgs []api.Message,
	plan PaddedLayoutConsumePlan,
) ([]int, bool) {
	if plan.Mode != PaddedLayoutConsumeMllamaImg || !plan.Active || tokenize == nil {
		return nil, false
	}
	return buildPaddedCompletionPromptTokensForSpans(ctx, tokenize, rendered, msgs, mllamaUserContentSpans(rendered), mllamaUserContentBounds)
}

func mllamaUserContentSpans(rendered string) []userContentSpan {
	var spans []userContentSpan
	searchFrom := 0
	for {
		rel := strings.Index(rendered[searchFrom:], mllamaUserStart)
		if rel < 0 {
			break
		}
		contentStart := searchFrom + rel + len(mllamaUserStart)
		// Skip optional newline after header (llama3 template).
		for contentStart < len(rendered) && rendered[contentStart] == '\n' {
			contentStart++
		}
		endRel := strings.Index(rendered[contentStart:], mllamaUserEnd)
		if endRel < 0 {
			break
		}
		contentEnd := contentStart + endRel
		spans = append(spans, userContentSpan{contentStart: contentStart, contentEnd: contentEnd})
		searchFrom = contentEnd + len(mllamaUserEnd)
	}
	return spans
}

func mllamaUserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := mllamaUserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}

func gemma4UserContentBounds(rendered string) (contentStart, contentEnd int, ok bool) {
	spans := gemma4UserContentSpans(rendered)
	if len(spans) == 0 {
		return 0, 0, false
	}
	last := spans[len(spans)-1]
	return last.contentStart, last.contentEnd, true
}
