package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/thinking"
)

const (
	glimmerGenerationPrefix = "<|start|>assistant"
	glimmerUserCommit       = " to=user<|message|>"
	lfm2GenerationSuffix    = "<|im_start|>assistant\n"
	lfm2EmptyThinkBlock     = "<think></think>"
)

func thinkExplicitlyOff(think *api.ThinkValue) bool {
	return think != nil && !think.Bool()
}

// ApplyNoThinkTailSuffix enforces think-off in the rendered prompt (mlx-serve
// chat.noThinkTailSuffix). Muse commits to=user when there are no tools.
// An unclosed <think> is closed from the bytes; lfm2-thinking gets an empty
// think block when the Go renderer did not open one.
func ApplyNoThinkTailSuffix(prompt, renderer string, tools []api.Tool, think *api.ThinkValue) string {
	if prompt == "" || !thinkExplicitlyOff(think) {
		return prompt
	}
	if len(tools) == 0 && strings.HasSuffix(prompt, glimmerGenerationPrefix) {
		return prompt + glimmerUserCommit
	}
	if thinking.PromptOpensThink(prompt) {
		return prompt + "</think>"
	}
	if strings.EqualFold(renderer, "lfm2-thinking") && strings.HasSuffix(prompt, lfm2GenerationSuffix) {
		return prompt + lfm2EmptyThinkBlock
	}
	return prompt
}
