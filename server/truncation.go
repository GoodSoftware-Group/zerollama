// Truncation fields on API responses.
//
// Why: runners and llama-server logged "truncating input prompt" / context-shift
// while clients got HTTP 200 with no signal that most of the prompt was dropped.
// Agents then treated prompt_eval_count ≈ num_ctx as a soft hint instead of an
// explicit overflow. These helpers copy runner + chatPrompt signals onto the
// final /api/chat and /api/generate chunks.
package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// applyPromptTruncation sets messages_* and prompt_* truncation fields on a chat response.
//
// Why chatOriginalTokens can override the runner: chatPrompt tail-truncates first
// (e.g. 44k → 8192), then the runner may trim one more token for slot headroom
// (8192 → 8191). The runner's OriginalPromptTokens would then be only 8192 —
// hiding the real pre-truncation size. Prefer the larger chatPrompt count.
func applyPromptTruncation(res *api.ChatResponse, cr llm.CompletionResponse, messagesDropped int, chatOriginalTokens int) {
	if messagesDropped > 0 {
		res.MessagesTruncated = true
		res.MessagesDropped = messagesDropped
	}
	if cr.PromptTruncated {
		res.PromptTruncated = true
		res.OriginalPromptTokens = cr.OriginalPromptTokens
	}
	if chatOriginalTokens > 0 && (!res.PromptTruncated || chatOriginalTokens > res.OriginalPromptTokens) {
		res.PromptTruncated = true
		res.OriginalPromptTokens = chatOriginalTokens
	}
}

// applyGenerateTruncation is the /api/generate counterpart of applyPromptTruncation.
// Why: same silent-200 problem on the generate path (including native template renders).
func applyGenerateTruncation(res *api.GenerateResponse, cr llm.CompletionResponse, messagesDropped int, chatOriginalTokens int) {
	if messagesDropped > 0 {
		res.MessagesTruncated = true
		res.MessagesDropped = messagesDropped
	}
	if cr.PromptTruncated {
		res.PromptTruncated = true
		res.OriginalPromptTokens = cr.OriginalPromptTokens
	}
	if chatOriginalTokens > 0 && (!res.PromptTruncated || chatOriginalTokens > res.OriginalPromptTokens) {
		res.PromptTruncated = true
		res.OriginalPromptTokens = chatOriginalTokens
	}
}
