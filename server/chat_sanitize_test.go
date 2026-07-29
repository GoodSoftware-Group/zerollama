package server

import (
	"strings"
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

func TestStripThinkToggleMarkers(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"OK /think", "OK"},
		{"OK /no_think", "OK"},
		{"OK /think /no_think", "OK"},
		{"OK", "OK"},
		{"Open src/think/main.py", "Open src/think/main.py"},
		{"Compare and/think versus", "Compare and/think versus"}, // not a trailing " /think"
		{"hi\n/think", "hi\n/think"},                             // newline-only form not used by Ollama inject
	}
	for _, tc := range cases {
		if got := stripThinkToggleMarkers(tc.in); got != tc.want {
			t.Fatalf("in=%q got=%q want=%q", tc.in, got, tc.want)
		}
	}
	if got := sanitizeAssistantContent("OK /think<|" + "im_end" + "|>"); got != "OK" {
		t.Fatalf("sanitize=%q", got)
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

func TestPreservePriorThinkingForRender(t *testing.T) {
	thinkOn := &api.ThinkValue{Value: true}
	msgs := []api.Message{
		{Role: "user", Content: "Step 1"},
		{Role: "assistant", Content: "Doing step 1.", Thinking: "MARKER_PRIOR"},
		{Role: "user", Content: "Step 2"},
	}
	got := preservePriorThinkingForRender(append([]api.Message(nil), msgs...), thinkOn)
	if !strings.Contains(got[1].Content, "MARKER_PRIOR") {
		t.Fatalf("prior thinking not embedded: %q", got[1].Content)
	}
	if !strings.Contains(got[1].Content, "<think>") {
		t.Fatalf("expected think tags: %q", got[1].Content)
	}

	// Current turn (assistant after last user) must not be rewritten.
	prefill := []api.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "partial", Thinking: "NOW"},
	}
	got = preservePriorThinkingForRender(prefill, thinkOn)
	if got[1].Content != "partial" {
		t.Fatalf("prefill mutated: %q", got[1].Content)
	}

	thinkOff := &api.ThinkValue{Value: false}
	fresh := []api.Message{
		{Role: "user", Content: "Step 1"},
		{Role: "assistant", Content: "Doing step 1.", Thinking: "MARKER_PRIOR"},
		{Role: "user", Content: "Step 2"},
	}
	got = preservePriorThinkingForRender(fresh, thinkOff)
	if strings.Contains(got[1].Content, "<think>") {
		t.Fatalf("think=false should not inject: %q", got[1].Content)
	}
}
