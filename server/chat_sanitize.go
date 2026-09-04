package server

import (
	"regexp"
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

// thinkToggleTrailRe matches template-injected Qwen think toggles that models
// often echo (minefield trap 66 mirror). Only trailing markers are stripped so
// paths like src/think/main.py are left alone.
var thinkToggleTrailRe = regexp.MustCompile(`(?i)(?:[ \t]+/(?:no_)?think)+\s*$`)

// stripThinkToggleMarkers removes trailing " /think" / " /no_think" from
// assistant content so harness scoring is not poisoned by template injection.
func stripThinkToggleMarkers(s string) string {
	return strings.TrimRight(thinkToggleTrailRe.ReplaceAllString(s, ""), " \t\n\r")
}

// sanitizeAssistantContent strips control tokens, think-toggle echo, leaked
// think tags, and unparsed tool markup (mlx-serve trimLeakedToolMarkup /
// trimTrailingThinkClosers).
func sanitizeAssistantContent(s string) string {
	return trimThinkTagLeaks(trimLeakedToolMarkup(stripThinkToggleMarkers(stripChatControlTokens(s))))
}

// sanitizeAssistantThinking applies the same tool-markup and think-tag cuts
// to reasoning so tags never ride out on that channel either.
func sanitizeAssistantThinking(s string) string {
	return trimThinkTagLeaks(trimLeakedToolMarkup(stripChatControlTokens(s)))
}

const thinkOpenTag = "<think>"
const thinkCloseTag = "</think>"

// trimThinkTagLeaks drops a pos-0 unclosed <think> and trailing </think>
// closers (mlx-serve). Mid-string tags are left alone.
func trimThinkTagLeaks(s string) string {
	lead := strings.TrimLeft(s, " \t\n\r")
	if strings.HasPrefix(lead, thinkOpenTag) {
		rest := lead[len(thinkOpenTag):]
		if !strings.Contains(rest, thinkCloseTag) {
			s = strings.TrimLeft(rest, " \t\n\r")
		}
	}
	for {
		t := strings.TrimRight(s, " \t\n\r")
		if !strings.HasSuffix(t, thinkCloseTag) {
			return t
		}
		s = t[:len(t)-len(thinkCloseTag)]
	}
}

var leakedToolOpeners = []string{
	"<tool_call>",
	"<|tool_call>",
	"<|tool_call|>",
	"<|tool_call_start|>",
	"<|tool_calls_section_begin|>",
	"<|tool_call_begin|>",
	"<atem:function_calls>",
	"<start_function_call>",
	"<function_calls>",
	"<|START_ACTION|>",
}

// trimLeakedToolMarkup cuts at the first tool-wrapper opener so unparsed
// markup never appears as assistant content. Applied once, at the earliest hit.
func trimLeakedToolMarkup(s string) string {
	cut := -1
	for _, o := range leakedToolOpeners {
		if i := strings.Index(s, o); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut < 0 {
		return s
	}
	return strings.TrimRight(s[:cut], " \t\n\r")
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
