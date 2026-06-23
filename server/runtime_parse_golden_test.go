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

func qwen3Model() *Model {
	return &Model{Config: model.ConfigV2{Parser: "qwen3"}}
}

func execCommandTool() api.Tools {
	return api.Tools{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "exec",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"command": {Type: api.PropertyType{"string"}},
				}),
			},
		},
	}}
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

// TestGoldenFunctionGemmaEmptyToolBlockThenContent — vLLM-class bug: empty tool
// blocks must not swallow trailing assistant content in streaming parse.
func TestGoldenFunctionGemmaEmptyToolBlockThenContent(t *testing.T) {
	t.Parallel()
	m := functionGemmaModel()
	tools := api.Tools{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "noop",
			Parameters: api.ToolFunctionParameters{Type: "object"},
		},
	}}
	chunks := []string{
		"<start_function_call>call:noop{}<end_function_call>",
		"Still here.",
	}

	id, _, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)

	var content string
	var finalCalls []api.ToolCall
	for i, ch := range chunks {
		done := i == len(chunks)-1
		c, _, calls, _, err := toolParseSessions.add(id, ch, done)
		if err != nil {
			t.Fatal(err)
		}
		content += c
		if len(calls) > 0 {
			finalCalls = calls
		}
	}
	if content != "Still here." {
		t.Fatalf("content %q", content)
	}
	if len(finalCalls) != 1 || finalCalls[0].Function.Name != "noop" {
		t.Fatalf("calls %+v", finalCalls)
	}
}

// TestGoldenFunctionGemmaAngleBracketsInContent — partial JSON / markup with '<'
// must not corrupt plain assistant text (Qwen3-class streaming bug).
func TestGoldenFunctionGemmaAngleBracketsInContent(t *testing.T) {
	t.Parallel()
	m := functionGemmaModel()
	tools := functionGemmaWeatherTool()
	id, _, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)

	chunks := []string{"a ", "<", "b", ">", " c"}
	var content string
	for i, ch := range chunks {
		c, _, _, _, err := toolParseSessions.add(id, ch, i == len(chunks)-1)
		if err != nil {
			t.Fatal(err)
		}
		content += c
	}
	if content != "a <b> c" {
		t.Fatalf("content %q", content)
	}
}

// TestGoldenQwen3AngleBracketsInToolArgumentsStreaming — '<' inside JSON tool args
// must survive chunk boundaries (Qwen3 / vLLM partial-JSON bug class).
func TestGoldenQwen3AngleBracketsInToolArgumentsStreaming(t *testing.T) {
	t.Parallel()
	m := qwen3Model()
	tools := execCommandTool()
	chunks := []string{
		`<tool_call>{"name":"exec","arguments":{"command":"ls && echo \"a > b and a `,
		`< b\""}}</tool_call>`,
	}

	id, method, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)
	if method != "qwen3" {
		t.Fatalf("method %q", method)
	}

	var finalCalls []api.ToolCall
	for i, ch := range chunks {
		_, _, calls, _, err := toolParseSessions.add(id, ch, i == len(chunks)-1)
		if err != nil {
			t.Fatal(err)
		}
		if len(calls) > 0 {
			finalCalls = calls
		}
	}
	if len(finalCalls) != 1 || finalCalls[0].Function.Name != "exec" {
		t.Fatalf("calls %+v", finalCalls)
	}
	cmd, ok := finalCalls[0].Function.Arguments.Get("command")
	if !ok || cmd != `ls && echo "a > b and a < b"` {
		t.Fatalf("command %v ok=%v", cmd, ok)
	}
}

// TestGoldenHarmonyToolThenFinalContent — tool commentary must not swallow a later
// final-channel assistant message in streaming parse (vLLM-class empty-block bug).
func TestGoldenHarmonyToolThenFinalContent(t *testing.T) {
	t.Parallel()
	m := harmonyModel()
	tools := functionGemmaWeatherTool()
	chunks := []string{
		`<|start|>assistant<|channel|>commentary to=functions.get_weather `,
		`<|constrain|>json<|message|>{"location":"Paris"}<|end|>`,
		`<|start|>assistant<|channel|>final<|message|>It is sunny in Paris.<|end|>`,
	}

	id, _, err := toolParseSessions.open(m, tools, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer toolParseSessions.close(id)

	var content string
	var finalCalls []api.ToolCall
	for i, ch := range chunks {
		done := i == len(chunks)-1
		c, _, calls, _, err := toolParseSessions.add(id, ch, done)
		if err != nil {
			t.Fatal(err)
		}
		content += c
		if len(calls) > 0 {
			finalCalls = calls
		}
	}
	if content != "It is sunny in Paris." {
		t.Fatalf("content %q", content)
	}
	if len(finalCalls) != 1 || finalCalls[0].Function.Name != "get_weather" {
		t.Fatalf("calls %+v", finalCalls)
	}
}
