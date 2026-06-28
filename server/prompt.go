package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)
type detokenizeFunc func(context.Context, []int) (string, error)

// chatPrompt accepts a list of messages and returns the prompt and images that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages.
// tokenBudget 0 uses chatPromptTokenBudget(opts). When detokenize is set and the rendered prompt still exceeds
// tokenBudget, tokens are dropped from the front (tail-keep) so single-message megaprompts fit the window.
//
// promptTokens is non-nil when tail-truncation ran, or for MLX when we captured
// ids for passthrough: routes pass CompletionRequest.PromptTokens so runners
// ingest exact IDs instead of re-tokenizing (avoids byte/special-token drift; MLX MTP).
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool, tokenBudget int, detokenize detokenizeFunc) (prompt string, images []llm.ImageData, messagesDropped int, promptTokens []int, err error) {
	// TODO: Ideally we would compute this from the projector metadata but some pieces are implementation dependent
	// Clip images are represented as 768 tokens, each an embedding
	imageNumTokens := 768

	currMsgIdx := 0

	if truncate {
		budget := tokenBudget
		if budget <= 0 {
			budget = chatPromptTokenBudget(opts)
		}
		if budget > 0 {
			var err error
			currMsgIdx, err = findChatPromptStartIdx(ctx, m, msgs, tools, think, tokenize, budget, imageNumTokens, m.ProjectorPaths != nil)
			if err != nil {
				return "", nil, 0, nil, err
			}
		}
	}

	if currMsgIdx > 0 {
		messagesDropped = currMsgIdx
		slog.Debug("truncating input messages which exceed context length", "truncated", messagesDropped)
	}

	for cnt, msg := range msgs[currMsgIdx:] {
		mmCount := len(msg.Images) + len(msg.PrecomputedEmbeddings) + len(msg.ProcessorOutputs)
		if slices.Contains(m.Config.ModelFamilies, "mllama") && mmCount > 1 {
			return "", nil, 0, nil, errors.New("this model only supports one image; more than one was requested (including multiple frames sampled from video)")
		}

		var prefix string
		prompt := msg.Content

		grids := modality.GridTHWPerRaster(msg)
		skipImgTags := modality.MessageSkipsVisionPlaceholdersForChat(msgs, currMsgIdx+cnt, true)
		images = modality.AppendPrecomputedImagesToLLM(msg, images)
		images = modality.AppendProcessorOutputsToLLM(msg, images)
		for idx, i := range msg.Images {
			imgData := llm.ImageData{
				ID:   len(images),
				Data: i,
			}
			if idx < len(grids) && len(grids[idx]) == 3 {
				imgData.GridTHW = append([]int(nil), grids[idx]...)
			}
			images = append(images, imgData)

			if m.Config.Renderer != "" || skipImgTags {
				continue
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}
		for _, a := range msg.AudioClips {
			imgData := llm.ImageData{
				ID:   len(images),
				Data: a,
			}
			images = append(images, imgData)

			if m.Config.Renderer != "" || skipImgTags {
				continue
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}
		msgs[currMsgIdx+cnt].Content = prefix + prompt
	}

	// truncate any messages that do not fit into the context window
	system := chatSystemPrefix(msgs, currMsgIdx)
	p, err := renderPrompt(m, append(system, msgs[currMsgIdx:]...), tools, think)
	if err != nil {
		return "", nil, 0, nil, err
	}

	if truncate && tokenize != nil {
		budget := tokenBudget
		if budget <= 0 {
			budget = chatPromptTokenBudget(opts)
		}
		if budget > 0 {
			p, _, promptTokens, err = tailTruncatePrompt(ctx, tokenize, detokenize, p, budget)
			if err != nil {
				return "", nil, 0, nil, err
			}
		}
	}

	// MLX Prepare re-tokenizes unless PromptTokens is set. Always capture ids once
	// so completion skips a second full encode on long agent prompts.
	if m.IsMLX() && len(promptTokens) == 0 && tokenize != nil && p != "" {
		ids, err := tokenize(ctx, p)
		if err != nil {
			return "", nil, 0, nil, err
		}
		promptTokens = ids
		slog.Debug("mlx prompt pre-tokenized for passthrough", "tokens", len(promptTokens))
	}

	return p, images, messagesDropped, promptTokens, nil
}

func tailTruncatePrompt(ctx context.Context, tokenize tokenizeFunc, detokenize detokenizeFunc, prompt string, budget int) (string, int, []int, error) {
	// Why token-ID drop: message-level truncate cannot shrink a single megaprompt;
	// detokenize is only for returning a string prompt to legacy paths — IDs are authoritative.
	if budget <= 0 || prompt == "" {
		return prompt, 0, nil, nil
	}
	ids, err := tokenize(ctx, prompt)
	if err != nil {
		return "", 0, nil, err
	}
	if len(ids) <= budget {
		return prompt, 0, ids, nil
	}
	dropped := len(ids) - budget
	kept := ids[dropped:]
	slog.Warn("prompt tail-truncated to fit context budget",
		"dropped_tokens", dropped,
		"kept_tokens", budget,
		"prompt_tokens", len(ids),
	)
	if detokenize == nil {
		return prompt, dropped, kept, nil
	}
	out, err := detokenize(ctx, kept)
	if err != nil {
		return "", dropped, kept, err
	}
	return out, dropped, kept, nil
}

// chatPromptLimits returns token budget and detokenize for chatPrompt when truncation is enabled.
func chatPromptLimits(m *Model, opts *api.Options, truncate bool, ctxLen int, detok detokenizeFunc) (int, detokenizeFunc) {
	if !truncate {
		return 0, nil
	}
	return effectiveChatPromptBudget(opts, m, ctxLen), detok
}

func chatSystemPrefix(msgs []api.Message, start int) []api.Message {
	var system []api.Message
	for j := range start {
		if msgs[j].Role == "system" {
			system = append(system, msgs[j])
		}
	}
	return system
}

func chatPromptTokenCount(msgs []api.Message, start, imageNumTokens int, hasImages bool) int {
	if !hasImages {
		return 0
	}
	n := 0
	for _, msg := range msgs[start:] {
		n += imageNumTokens * len(msg.Images)
	}
	return n
}

// findChatPromptStartIdx binary-searches the earliest message index that fits in budget.
// Always includes the last message even when over budget.
func findChatPromptStartIdx(
	ctx context.Context,
	m *Model,
	msgs []api.Message,
	tools []api.Tool,
	think *api.ThinkValue,
	tokenize tokenizeFunc,
	budget int,
	imageNumTokens int,
	hasImages bool,
) (int, error) {
	lastIdx := len(msgs) - 1
	if lastIdx < 0 || budget <= 0 {
		return 0, nil
	}

	fits := func(start int) (bool, error) {
		system := chatSystemPrefix(msgs, start)
		p, err := renderPrompt(m, append(system, msgs[start:]...), tools, think)
		if err != nil {
			return false, err
		}
		s, err := tokenize(ctx, p)
		if err != nil {
			return false, err
		}
		ctxLen := len(s) + chatPromptTokenCount(msgs, start, imageNumTokens, hasImages)
		return ctxLen <= budget, nil
	}

	if ok, err := fits(0); err != nil {
		return 0, err
	} else if ok {
		return 0, nil
	}

	lo, hi := 1, lastIdx
	best := lastIdx
	for lo <= hi {
		mid := (lo + hi) / 2
		ok, err := fits(mid)
		if err != nil {
			return 0, err
		}
		if ok {
			best = mid
			hi = mid - 1
		} else {
			lo = mid + 1
		}
	}
	return best, nil
}

func renderPrompt(m *Model, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if m.Config.Renderer != "" {
		rendererName := resolveRendererName(m)
		rendered, err := renderers.RenderWithRenderer(rendererName, msgs, tools, think)
		if err != nil {
			return "", err
		}
		return rendered, nil
	}

	var b bytes.Buffer
	thinkVal := false
	thinkLevel := ""
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	}
	if err := m.Template.Execute(&b, template.Values{Messages: msgs, Tools: tools, Think: thinkVal, ThinkLevel: thinkLevel, IsThinkSet: think != nil}); err != nil {
		return "", err
	}
	return b.String(), nil
}

func imageTaggedMessages(m *Model, msgs []api.Message, start int, clearImages bool) ([]api.Message, []llm.ImageData, error) {
	renderMsgs := slices.Clone(msgs)
	var images []llm.ImageData

	for cnt, msg := range renderMsgs[start:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return nil, nil, errors.New("this model only supports one image while more than one image requested")
		}

		var prefix string
		prompt := msg.Content

		for _, i := range msg.Images {
			imgData := llm.ImageData{
				ID:   len(images),
				Data: i,
			}
			images = append(images, imgData)

			if m.Config.Renderer != "" {
				continue
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}

		if m.Config.Renderer == "" {
			renderMsgs[start+cnt].Content = prefix + prompt
		}
		if clearImages {
			renderMsgs[start+cnt].Images = nil
		}
	}

	return renderMsgs, images, nil
}
