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
	}, 2, 0)

	if !res.PromptTruncated || res.OriginalPromptTokens != 14841 {
		t.Fatalf("prompt truncation: %+v", res)
	}
	if !res.MessagesTruncated || res.MessagesDropped != 2 {
		t.Fatalf("messages truncation: %+v", res)
	}
}

func TestApplyPromptTruncationChatOriginalTokens(t *testing.T) {
	res := api.ChatResponse{}
	applyPromptTruncation(&res, llm.CompletionResponse{}, 0, 44000)

	if !res.PromptTruncated {
		t.Fatal("expected prompt_truncated=true from chatOriginalTokens")
	}
	if res.OriginalPromptTokens != 44000 {
		t.Fatalf("expected original_prompt_tokens=44000, got %d", res.OriginalPromptTokens)
	}
}

func TestApplyPromptTruncationChatOverridesSmaller(t *testing.T) {
	res := api.ChatResponse{}
	applyPromptTruncation(&res, llm.CompletionResponse{
		PromptTruncated:      true,
		OriginalPromptTokens: 8192,
	}, 0, 44000)

	if res.OriginalPromptTokens != 44000 {
		t.Fatalf("chatOriginalTokens should win over smaller runner value, got %d", res.OriginalPromptTokens)
	}
}

func TestApplyGenerateTruncationChatOriginalTokens(t *testing.T) {
	res := api.GenerateResponse{}
	applyGenerateTruncation(&res, llm.CompletionResponse{}, 0, 44000)

	if !res.PromptTruncated {
		t.Fatal("expected prompt_truncated=true")
	}
	if res.OriginalPromptTokens != 44000 {
		t.Fatalf("expected 44000, got %d", res.OriginalPromptTokens)
	}
}
