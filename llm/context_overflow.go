package llm

import "fmt"

// ContextOverflowMessage names both the prompt size and the model window
// (mlx-serve contextOverflowMessage). Clients can render a card without parsing.
func ContextOverflowMessage(promptTokens, contextLength int) string {
	return fmt.Sprintf("input length (%d tokens) exceeds the model's maximum context length (%d tokens)", promptTokens, contextLength)
}
