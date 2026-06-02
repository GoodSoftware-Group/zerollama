package server

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestChatNeedsLegacyRunner(t *testing.T) {
	t.Parallel()
	plain := api.ChatRequest{
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}
	if chatNeedsLegacyRunner(plain.Messages, plain) {
		t.Fatal("plain text should use runtime")
	}

	tools := plain
	tools.Tools = api.Tools{{Type: "function", Function: api.ToolFunction{Name: "f"}}}
	if chatNeedsLegacyRunner(tools.Messages, tools) {
		t.Fatal("tools should use runtime when other legacy features absent")
	}

	logprobs := plain
	logprobs.Logprobs = true
	if !chatNeedsLegacyRunner(logprobs.Messages, logprobs) {
		t.Fatal("logprobs require legacy")
	}

	think := plain
	think.Think = &api.ThinkValue{Value: true}
	if !chatNeedsLegacyRunner(think.Messages, think) {
		t.Fatal("think require legacy")
	}

	vision := plain
	vision.Messages = []api.Message{{
		Role:    "user",
		Content: "look",
		Images:  []api.ImageData{[]byte{1}},
	}}
	if !chatNeedsLegacyRunner(vision.Messages, vision) {
		t.Fatal("images require legacy")
	}

	toolMsg := plain
	toolMsg.Messages = []api.Message{{
		Role: "assistant",
		ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: "f"},
		}},
	}}
	if chatNeedsLegacyRunner(toolMsg.Messages, toolMsg) {
		t.Fatal("tool_calls history should use runtime")
	}
}
