package server

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// guessParserForArchitecture maps GGUF arch (+ optional chat template hints) to a
// builtin parser name when the manifest omits parser. Why: tool/thinking stream
// parsing is family-specific; wrong parser breaks agent tool calls even when the
// model weights are correct (LocalAI ModelMetadata discovery pattern).
func guessParserForArchitecture(arch, chatTemplate string) string {
	arch = strings.ToLower(strings.TrimSpace(arch))
	tmpl := strings.ToLower(chatTemplate)
	thinking := strings.Contains(tmpl, "thinking") ||
		strings.Contains(tmpl, "<thought") ||
		strings.Contains(tmpl, "redacted_thinking")

	switch arch {
	case "qwen35", "qwen35moe":
		return "qwen3.5"
	case "qwen3", "qwen3moe":
		if thinking {
			return "qwen3-thinking"
		}
		return "qwen3"
	case "qwen3vl", "qwen3vlmoe":
		if thinking {
			return "qwen3-vl-thinking"
		}
		return "qwen3-vl-instruct"
	case "gemma4":
		if thinking {
			return "gemma4"
		}
		return "gemma4-no-thinking"
	case "gptoss", "gpt_oss", "gpt-oss":
		return "harmony"
	case "lfm2", "lfm2moe":
		if thinking {
			return "lfm2-thinking"
		}
		return "lfm2"
	case "ministral", "ministral3":
		return "ministral"
	case "deepseek2", "deepseek3":
		return "deepseek3"
	case "glm4", "glm4moe":
		return "glm-4.7"
	case "olmo3":
		if thinking {
			return "olmo3-think"
		}
		return "olmo3"
	case "nemotron3", "nemotron3nano":
		return "nemotron-3-nano"
	case "functiongemma":
		return "functiongemma"
	case "cogito":
		return "cogito"
	default:
		return ""
	}
}

func guessParserFromKV(kv ggml.KV) string {
	if kv == nil {
		return ""
	}
	return guessParserForArchitecture(kv.Architecture(), kv.ChatTemplate())
}
