package server

import (
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

// goldenTokenize counts whitespace-separated words (stable, deterministic for parity tests).
func goldenTokenize(_ context.Context, s string) ([]int, error) {
	parts := strings.Fields(s)
	out := make([]int, len(parts))
	for i := range parts {
		out[i] = i
	}
	return out, nil
}

func testRenderTemplate(t *testing.T) *template.Template {
	t.Helper()
	tmpl, err := template.Parse(`
{{- if .System }}{{ .System }}
{{- end }}
{{- range .Messages }}
{{ .Role }}: {{ .Content }}
{{- end }}`)
	if err != nil {
		t.Fatal(err)
	}
	return tmpl
}

// TestGoldenRenderChatMatchesChatPromptNoTruncate ensures /internal/render-chat
// (Phase 12) produces the same prompt as ggml chatPrompt when truncation is off.
func TestGoldenRenderChatMatchesChatPromptNoTruncate(t *testing.T) {
	t.Parallel()
	tmpl := testRenderTemplate(t)
	m := &Model{
		Template: tmpl,
		System:   "You are helpful.",
	}
	raw := []api.Message{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", Content: "Checking."},
		{Role: "user", Content: "Paris"},
	}
	prep := prepareRenderMessages(m, raw)
	opts := api.Options{Runner: api.Runner{NumCtx: 8192}}

	promptChat, _, _, _, err := chatPrompt(
		context.Background(), m, goldenTokenize, &opts, prep, nil, nil, false, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{}}}
	promptRender, mode, dropped, _, err := s.renderChatPromptPrepared(
		context.Background(), m, prep, nil, nil, 0, 0, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "none" || dropped {
		t.Fatalf("mode=%q dropped=%v", mode, dropped)
	}
	if diff := cmp.Diff(promptRender, promptChat); diff != "" {
		t.Fatalf("render-chat vs chatPrompt (-render +chat):\n%s", diff)
	}
}

// TestGoldenRenderChatTokenizedTruncationMatchesChatPrompt checks ggml chatPrompt and
// render-chat use the same prompt token budget when num_predict is set (chatPromptTokenBudget).
func TestGoldenRenderChatTokenizedTruncationMatchesChatPrompt(t *testing.T) {
	t.Parallel()
	tmpl := testRenderTemplate(t)
	m := &Model{
		Template: tmpl,
		Name:     "golden-trunc",
	}
	raw := []api.Message{
		{Role: "user", Content: "one two three four five six seven"},
		{Role: "assistant", Content: "eight nine ten eleven twelve"},
		{Role: "user", Content: "thirteen fourteen"},
	}
	prep := prepareRenderMessages(m, raw)

	numCtx := 32
	numPredict := 28
	opts := api.Options{Runner: api.Runner{NumCtx: numCtx}, NumPredict: numPredict}

	promptChat, _, _, _, err := chatPrompt(
		context.Background(), m, goldenTokenize, &opts, prep, nil, nil, true, 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{}}}
	key := schedulerModelKey(m)
	s.sched.loaded[key] = &runnerRef{
		llama: &mockRunner{},
		model: m,
	}

	promptRender, mode, dropped, _, err := s.renderChatPromptPrepared(
		context.Background(), m, prep, nil, nil, numCtx, numPredict, true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "tokenize" {
		t.Fatalf("mode=%q want tokenize", mode)
	}
	if !dropped {
		t.Fatal("expected prefix drop under tight budget")
	}
	if diff := cmp.Diff(promptRender, promptChat); diff != "" {
		t.Fatalf("tokenized truncation parity (-render +chat):\n%s", diff)
	}
}

// TestGoldenRenderChatToolsTemplateSnapshot locks a Modelfile-style tools prompt shape.
func TestGoldenRenderChatToolsTemplateSnapshot(t *testing.T) {
	t.Parallel()
	tmpl, err := template.Parse(`{{- if .Tools }}TOOLS:
{{- range .Tools }}
- {{ .Function.Name }}
{{- end }}
{{- end }}
{{- range .Messages }}{{ .Role }}: {{ .Content }}
{{- end }}`)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{
		Template: tmpl,
		Config:   model.ConfigV2{Parser: "functiongemma"},
	}
	raw := []api.Message{{Role: "user", Content: "Call weather for Paris"}}
	tools := api.Tools{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "weather",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"city": {Type: api.PropertyType{"string"}},
				}),
			},
		},
	}}
	prep := prepareRenderMessages(m, raw)
	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{}}}
	prompt, mode, dropped, _, err := s.renderChatPromptPrepared(
		context.Background(), m, prep, tools, nil, 0, 0, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "none" || dropped {
		t.Fatalf("mode=%q dropped=%v", mode, dropped)
	}
	const want = "TOOLS:\n- get_weatheruser: Call weather for Paris"
	if diff := cmp.Diff(strings.TrimSpace(prompt), strings.TrimSpace(want)); diff != "" {
		t.Fatalf("tools template golden (-got +want):\n%s", diff)
	}
}
