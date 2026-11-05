package modelrepair

import "strings"

// suggestedFamilyChatStack maps GGUF architecture / parser hints to a builtin
// RENDERER+PARSER pair that is safe to auto-install for passthrough templates.
// ok=false means we have no known stack (manual review only).
func suggestedFamilyChatStack(show ShowInfo) (renderer, parser string, ok bool) {
	arch := strings.ToLower(strings.TrimSpace(show.Architecture))
	switch arch {
	case "deepseek4", "deepseek3", "deepseek2":
		return "deepseek3", "deepseek3", true
	case "qwen35", "qwen35moe":
		return "qwen3.5", "qwen3.5", true
	case "gemma4":
		return "gemma4", "gemma4", true
	case "gptoss", "gpt_oss", "gpt-oss":
		return "harmony", "harmony", true
	}
	p := strings.ToLower(strings.TrimSpace(show.Parser))
	switch {
	case strings.HasPrefix(p, "deepseek"):
		return "deepseek3", "deepseek3", true
	case strings.HasPrefix(p, "qwen3.5") || p == "qwen35":
		return "qwen3.5", "qwen3.5", true
	case strings.HasPrefix(p, "gemma4"):
		return "gemma4", "gemma4", true
	case p == "harmony":
		return "harmony", "harmony", true
	}
	return "", "", false
}
