package api

import "strings"

// RequiredToolCallHint is appended to the last user turn when the client
// asked for a forced tool call (OpenAI tool_choice required/named, Anthropic any/tool).
// Prompt-side only — we do not constrain decode to a grammar.
const RequiredToolCallHint = "You must invoke at least one tool this turn."

func ToolChoiceRequiresCall(kind string) bool {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "required", "any":
		return true
	default:
		return false
	}
}

// AppendRequiredToolCallHint adds RequiredToolCallHint to the last user message
// when force is set and tools remain. Idempotent if the hint is already present.
func AppendRequiredToolCallHint(msgs []Message, tools []Tool, force bool) []Message {
	if !force || len(tools) == 0 {
		return msgs
	}
	return appendLastUserParagraph(msgs, RequiredToolCallHint)
}

func appendLastUserParagraph(msgs []Message, para string) []Message {
	if para == "" || len(msgs) == 0 {
		return msgs
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		if !strings.EqualFold(msgs[i].Role, "user") {
			continue
		}
		if strings.Contains(msgs[i].Content, para) {
			return msgs
		}
		c := strings.TrimRight(msgs[i].Content, " \t\r\n")
		if c == "" {
			msgs[i].Content = para
		} else {
			msgs[i].Content = c + "\n\n" + para
		}
		return msgs
	}
	return msgs
}
