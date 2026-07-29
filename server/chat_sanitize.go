package server

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

func chatControlTokens() []string {
	return []string{
		"<|" + "endoftext" + "|>",
		"<|" + "im_start" + "|>",
		"<|" + "im_end" + "|>",
		"<|" + "redacted_im_start" + "|>",
		"<|" + "redacted_im_end" + "|>",
	}
}

// stripChatControlTokens removes leaked chat-template tokens from model output.
func stripChatControlTokens(s string) string {
	for _, marker := range chatControlTokens() {
		if idx := strings.Index(s, marker); idx >= 0 {
			s = s[:idx]
		}
	}
	return strings.TrimRight(s, " \t\n\r")
}

func usesQwenStyleChat(m *Model) bool {
	if m == nil {
		return false
	}
	if slices.Contains([]string{"qwen35", "qwen35moe", "qwen3"}, m.PrimaryFamily()) {
		return true
	}
	return model.ParseName(m.Name).Model == "deepseek-r1"
}

// filterThinkTags strips thinking traces and leaked control tokens from prior
// assistant turns so multi-turn chat stays well-formed.
func filterThinkTags(msgs []api.Message, m *Model) []api.Message {
	if !usesQwenStyleChat(m) {
		return msgs
	}

	finalUserIndex := -1
	for i, msg := range msgs {
		if msg.Role == "user" {
			finalUserIndex = i
		}
	}

	for i, msg := range msgs {
		if msg.Role != "assistant" || i >= finalUserIndex {
			continue
		}
		thinkingState := &thinking.Parser{
			OpeningTag: "<think>",
			ClosingTag: "</think>",
		}
		_, content := thinkingState.AddContent(msg.Content)
		msgs[i].Content = stripChatControlTokens(content)
	}
	return msgs
}

// preservePriorThinkingForRender re-embeds resent message.Thinking into prior
// assistant Content after filterThinkTags. Stock Go templates (e.g. qwen3) only
// emit .Thinking after the last user index, so multi-turn studies that correctly
// resend thinking still assemble a history the model reads as "I never thought"
// (minefield trap 04). Injection uses <think>…</think> so Content-emitting
// templates surface the marker without requiring a Modelfile rewrite.
func preservePriorThinkingForRender(msgs []api.Message, think *api.ThinkValue) []api.Message {
	if think == nil || !think.Bool() || len(msgs) == 0 {
		return msgs
	}
	lastUser := -1
	for i, m := range msgs {
		if m.Role == "user" {
			lastUser = i
		}
	}
	for i, m := range msgs {
		if m.Role != "assistant" {
			continue
		}
		th := strings.TrimSpace(m.Thinking)
		if th == "" {
			continue
		}
		// Tool-call turns keep historical thinking stripped (Qwen tool-loop templates).
		if len(m.ToolCalls) > 0 {
			continue
		}
		// Current assistant turn (after last user, or trailing assistant prefill):
		// the template / renderer already owns .Thinking.
		if lastUser >= 0 && i >= lastUser {
			continue
		}
		if lastUser < 0 {
			continue
		}
		if strings.Contains(m.Content, th) {
			continue
		}
		msgs[i].Content = "<think>" + m.Thinking + "</think>\n" + m.Content
	}
	return msgs
}
