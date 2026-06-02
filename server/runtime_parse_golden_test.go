package server

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func functionGemmaWeatherTool() api.Tools {
	return api.Tools{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "get_weather",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"city": {Type: api.PropertyType{"string"}},
				}),
			},
		},
	}}
}

func functionGemmaModel() *Model {
	return &Model{Config: model.ConfigV2{Parser: "functiongemma"}}
}

func harmonyModel() *Model {
	return &Model{Config: model.ConfigV2{Parser: "harmony"}}
}

// TestGoldenHarmonyParseToolOutput ensures harmony builtin parser works via tool parse sessions.
func TestGoldenHarmonyParseToolOutput(t *testing.T) {
	t.Parallel()
	m := harmonyModel()
	tools := functionGemmaWeatherTool()
	// Omit trailing <|call|> in one-shot; harmony drains JSON at done (see harmony/harmonyparser_test.go).
	raw := `<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|end|>`

	id, method, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)
	if method != "harmony" {
		t.Fatalf("method %q", method)
	}

	_, _, calls, methodOut, err := toolParseSessions.add(id, raw, true)
	if err != nil {
		t.Fatal(err)
	}
	if methodOut != "harmony" {
		t.Fatalf("methodOut %q", methodOut)
	}
	if len(calls) != 1 {
		t.Fatalf("calls %+v", calls)
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name %q", calls[0].Function.Name)
	}
	args := calls[0].Function.Arguments.ToMap()
	if args["location"] != "San Francisco" {
		t.Fatalf("args %v", args)
	}
}

// TestGoldenFunctionGemmaParseToolOutput ensures /internal/parse-tool-output session
// path matches model/parsers/functiongemma for a one-shot tool call (Phase 12).
func TestGoldenFunctionGemmaParseToolOutput(t *testing.T) {
	t.Parallel()
	m := functionGemmaModel()
	tools := functionGemmaWeatherTool()
	raw := "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>"

	id, method, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)
	if method != "functiongemma" {
		t.Fatalf("method %q", method)
	}

	content, thinking, calls, methodOut, err := toolParseSessions.add(id, raw, true)
	if err != nil {
		t.Fatal(err)
	}
	if thinking != "" || content != "" {
		t.Fatalf("content=%q thinking=%q", content, thinking)
	}
	if methodOut != "functiongemma" {
		t.Fatalf("methodOut %q", methodOut)
	}
	want := []api.ToolCall{{
		Function: api.ToolCallFunction{
			Name:      "get_weather",
			Arguments: testArgs(map[string]any{"city": "Paris"}),
		},
	}}
	if diff := cmp.Diff(calls, want, argsComparer); diff != "" {
		t.Fatalf("tool_calls (-got +want):\n%s", diff)
	}
}

// TestGoldenFunctionGemmaParseStreamingChunks matches chunked streaming parse used by
// /internal/parse-tool-output/chunk (same chunk split as model/parsers/functiongemma_test).
func TestGoldenFunctionGemmaParseStreamingChunks(t *testing.T) {
	t.Parallel()
	m := functionGemmaModel()
	tools := functionGemmaWeatherTool()
	chunks := []string{
		"<", "start", "_", "function", "_", "call", ">",
		"call", ":", "get", "_", "weather", "{",
		"city", ":", "<", "escape", ">", "Paris", "<", "escape", ">",
		"}", "<", "end", "_", "function", "_", "call", ">",
	}

	id, _, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)

	var finalCalls []api.ToolCall
	for i, ch := range chunks {
		done := i == len(chunks)-1
		_, _, calls, _, err := toolParseSessions.add(id, ch, done)
		if err != nil {
			t.Fatal(err)
		}
		if len(calls) > 0 {
			finalCalls = calls
		}
	}
	want := []api.ToolCall{{
		Function: api.ToolCallFunction{
			Name:      "get_weather",
			Arguments: testArgs(map[string]any{"city": "Paris"}),
		},
	}}
	if diff := cmp.Diff(finalCalls, want, argsComparer); diff != "" {
		t.Fatalf("streamed tool_calls (-got +want):\n%s", diff)
	}
}

// TestGoldenFunctionGemmaRenderParseRoundtrip checks builtin parser + non-empty render;
// does not assert model output matches prompt (see ops golden on real weights).
func TestGoldenFunctionGemmaRenderParseRoundtrip(t *testing.T) {
	t.Parallel()
	tmpl, err := template.Parse(`{{- range .Messages }}{{ .Content }}{{- end }}`)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{
		Template: tmpl,
		Config:   model.ConfigV2{Parser: "functiongemma"},
	}
	msgs := []api.Message{{Role: "user", Content: "weather in Paris"}}
	tools := functionGemmaWeatherTool()
	prep := prepareRenderMessages(m, msgs)

	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{}}}
	prompt, mode, dropped, hasSupport, err := s.renderChatPromptPrepared(
		t.Context(), m, prep, tools, nil, 0, 0, false,
	)
	if err != nil || mode != "none" || dropped {
		t.Fatalf("render: mode=%q dropped=%v err=%v", mode, dropped, err)
	}
	if prompt == "" || !hasSupport {
		t.Fatalf("prompt=%q hasSupport=%v", prompt, hasSupport)
	}

	modelOut := "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>"
	id, _, err := toolParseSessions.open(m, tools, prep, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)
	_, _, calls, _, err := toolParseSessions.add(id, modelOut, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("calls %+v", calls)
	}
}
