package template

import (
	"strings"

	"github.com/ollama/ollama/api"
)

// ContinueFinalAllowed is mlx-serve's continuable check: last turn is
// assistant text, not a tool-call reply.
func ContinueFinalAllowed(msgs []api.Message) bool {
	if len(msgs) == 0 {
		return false
	}
	last := msgs[len(msgs)-1]
	if last.Role != "assistant" {
		return false
	}
	if len(last.ToolCalls) > 0 {
		return false
	}
	return strings.TrimSpace(last.Content) != ""
}

// ApplyContinueFinal trims the rendered prompt so it ends on the last
// assistant text (mid-turn prefill). Generation prompt and close tags after
// that content are dropped.
func ApplyContinueFinal(prompt string, msgs []api.Message) (string, bool) {
	if !ContinueFinalAllowed(msgs) {
		return prompt, false
	}
	content := strings.TrimRight(msgs[len(msgs)-1].Content, " \t\r\n")
	if content == "" {
		return prompt, false
	}
	prompt = stripTrainGenerationPrompt(prompt)
	idx := strings.LastIndex(prompt, content)
	if idx < 0 {
		return prompt, false
	}
	return prompt[:idx+len(content)], true
}
