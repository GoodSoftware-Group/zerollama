package server

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestModelThinkingDiscovery(t *testing.T) {
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	for _, tt := range []struct {
		name string
		m    Model
		want *model.Thinking
	}{
		{"gemma4", Model{Config: model.ConfigV2{Renderer: "gemma4", Parser: "gemma4", Capabilities: []string{"completion", "thinking"}}}, &model.Thinking{Values: []any{false, true}, Default: true}},
		{"qwen38", Model{Config: model.ConfigV2{Renderer: "qwen3.8", Parser: "qwen3.5", Capabilities: []string{"completion", "thinking"}}}, &model.Thinking{Values: []any{false, "low", "medium", "xhigh"}, Default: "xhigh"}},
		{"no renderer", Model{}, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.m.Thinking()
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %#v, want %#v", got, tt.want)
			}
		})
	}
}
