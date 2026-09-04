package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestApplyNoThinkTailSuffix(t *testing.T) {
	off := &api.ThinkValue{Value: false}
	on := &api.ThinkValue{Value: true}
	weather := []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "get_weather"}}}

	t.Run("glimmer commits user channel", func(t *testing.T) {
		in := "<|begin_of_text|><|start|>user<|message|>Hi<|eot|><|start|>assistant"
		got := ApplyNoThinkTailSuffix(in, "glimmer", nil, off)
		if got != in+" to=user<|message|>" {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("glimmer keeps tools path", func(t *testing.T) {
		in := "<|start|>assistant"
		got := ApplyNoThinkTailSuffix(in, "glimmer", weather, off)
		if got != in {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("think on unchanged", func(t *testing.T) {
		in := "<|start|>assistant"
		if got := ApplyNoThinkTailSuffix(in, "glimmer", nil, on); got != in {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("closes unclosed think", func(t *testing.T) {
		in := "<|im_start|>assistant\n<think>"
		got := ApplyNoThinkTailSuffix(in, "", nil, off)
		if got != in+"</think>" {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("lfm2-thinking empty block", func(t *testing.T) {
		in := "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
		got := ApplyNoThinkTailSuffix(in, "lfm2-thinking", nil, off)
		if got != in+"<think></think>" {
			t.Fatalf("got %q", got)
		}
	})
}
