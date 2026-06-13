// Truncation fields on API responses — why: runners logged "truncating input prompt"
// but clients got HTTP 200 with no signal that most of the prompt was dropped.
package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func applyPromptTruncation(res *api.ChatResponse, cr llm.CompletionResponse, messagesDropped int) {
	if messagesDropped > 0 {
		res.MessagesTruncated = true
		res.MessagesDropped = messagesDropped
	}
	if cr.PromptTruncated {
		res.PromptTruncated = true
		res.OriginalPromptTokens = cr.OriginalPromptTokens
	}
}

func applyGenerateTruncation(res *api.GenerateResponse, cr llm.CompletionResponse, messagesDropped int) {
	if messagesDropped > 0 {
		res.MessagesTruncated = true
		res.MessagesDropped = messagesDropped
	}
	if cr.PromptTruncated {
		res.PromptTruncated = true
		res.OriginalPromptTokens = cr.OriginalPromptTokens
	}
}
