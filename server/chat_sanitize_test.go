package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestStripChatControlTokens(t *testing.T) {
	in := "Hello!<|" + "endoftext" + "|><|" + "im_start" + "|>user\nhow"
	got := stripChatControlTokens(in)
	if got != "Hello!" {
		t.Fatalf("got %q", got)
	}
}

func TestDefaultRendererForFamilyQwen35(t *testing.T) {
	got := defaultRendererForFamily(&Model{Config: model.ConfigV2{ModelFamily: "qwen35"}})
	if got != "qwen3.5" {
		t.Fatalf("got %q", got)
	}
}

func TestFilterThinkTagsStripsControlTokensFromHistory(t *testing.T) {
	m := &Model{Config: model.ConfigV2{ModelFamily: "qwen35"}}
	msgs := []api.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "Hello!<|" + "endoftext" + "|><|" + "im_start" + "|>user"},
		{Role: "user", Content: "again"},
	}
	got := filterThinkTags(msgs, m)
	if got[1].Content != "Hello!" {
		t.Fatalf("got %q", got[1].Content)
	}
}
