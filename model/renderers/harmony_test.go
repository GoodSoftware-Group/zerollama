package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestHarmonyRendererBasicChat(t *testing.T) {
	r := &HarmonyRenderer{}
	got, err := r.Render([]api.Message{
		{Role: "user", Content: "Say hello in one word."},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "<|start|>user<|message|>Say hello in one word.<|end|>") {
		t.Fatalf("missing user message: %q", got)
	}
	if !strings.HasSuffix(got, "<|start|>assistant") {
		t.Fatalf("expected generation prefix, got %q", got)
	}
}

func TestHarmonyRendererSystemAndTools(t *testing.T) {
	r := &HarmonyRenderer{}
	got, err := r.Render([]api.Message{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Weather?"},
	}, []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"location"},
				Properties: func() *api.ToolPropertiesMap {
					m := api.NewToolPropertiesMap()
					m.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "City"})
					return m
				}(),
			},
		},
	}}, &api.ThinkValue{Value: "high"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "Reasoning: high") {
		t.Fatalf("missing reasoning level: %q", got)
	}
	if !strings.Contains(got, "Be concise.") {
		t.Fatalf("missing instructions: %q", got)
	}
	if !strings.Contains(got, "type get_weather") {
		t.Fatalf("missing tool declaration: %q", got)
	}
}
