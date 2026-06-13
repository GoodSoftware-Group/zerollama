package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestApplyPromptTruncation(t *testing.T) {
	res := api.ChatResponse{}
	applyPromptTruncation(&res, llm.CompletionResponse{
		PromptTruncated:      true,
		OriginalPromptTokens: 14841,
	}, 2)

	if !res.PromptTruncated || res.OriginalPromptTokens != 14841 {
		t.Fatalf("prompt truncation: %+v", res)
	}
	if !res.MessagesTruncated || res.MessagesDropped != 2 {
		t.Fatalf("messages truncation: %+v", res)
	}
}
