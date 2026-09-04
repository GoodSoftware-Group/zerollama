package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen35ParserXMLToolCall(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						props.Set("days", api.ToolProperty{Type: api.PropertyType{"integer"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: false})
	input := "<tool_call><function=get_weather><parameter=location>\nSan Francisco\n</parameter><parameter=days>\n3\n</parameter></function></tool_call>"
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}

	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "San Francisco" {
		t.Fatalf("expected location %q, got %v", "San Francisco", location)
	}

	days, ok := calls[0].Function.Arguments.Get("days")
	if !ok || days != 3 {
		t.Fatalf("expected days %d, got %v", 3, days)
	}
}

func TestQwen35ParserMissingToolCloseOnDone(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}
	parser.Init(tools, nil, &api.ThinkValue{Value: false})

	body := "<tool_call><function=get_weather><parameter=location>\nParis\n</parameter></function>"
	content, thinking, calls, err := parser.Add(body, false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" || thinking != "" || len(calls) != 0 {
		t.Fatalf("expected no emit before done, got content=%q thinking=%q calls=%d", content, thinking, len(calls))
	}

	content, thinking, calls, err = parser.Add("", true)
	if err != nil {
		t.Fatalf("parse failed on done: %v", err)
	}
	if content != "" || thinking != "" {
		t.Fatalf("expected empty content/thinking, got %q / %q", content, thinking)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected get_weather, got %#v", calls)
	}
	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "Paris" {
		t.Fatalf("expected location Paris, got %v", location)
	}
}

func TestQwen35ParserPluralToolCallsWrapper(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}
	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}
	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	input := "Plan<tool_calls:0><function=get_weather><parameter=location>\nLisbon\n</parameter></function></tool_calls>"
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "Plan" {
		t.Fatalf("expected thinking Plan, got %q", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected get_weather, got %#v", calls)
	}
	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "Lisbon" {
		t.Fatalf("expected Lisbon, got %v", location)
	}
}

func TestQwen35ParserThinkingWithExplicitOpeningTag(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("<think>\nLet me think...</think>Answer.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "Let me think..." {
		t.Fatalf("expected thinking %q, got %q", "Let me think...", thinking)
	}
	if content != "Answer." {
		t.Fatalf("expected content %q, got %q", "Answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserAssistantPrefillStartsInContent(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	last := &api.Message{Role: "assistant", Content: "Prefilled response start"}
	parser.Init(nil, last, nil)

	content, thinking, calls, err := parser.Add(" and continued", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected no thinking for assistant prefill continuation, got %q", thinking)
	}
	if content != " and continued" {
		t.Fatalf("expected content %q, got %q", " and continued", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserToolCallEmittedInThinkingIsParsed(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	input := `Need weather lookup<tool_call><function=get_weather><parameter=location>
SF
</parameter></function></tool_call>`
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}

	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "SF" {
		t.Fatalf("expected location %q, got %v", "SF", location)
	}
}

func TestQwen35ParserToolCallEmittedInThinkingIsParsedWhenToolCallTagIsSplitAcrossChunks(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Need weather lookup<tool_c", false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in first chunk, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add(`all><function=get_weather><parameter=location>
SF
</parameter></function></tool_call>`, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking, got %q", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}

	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "SF" {
		t.Fatalf("expected location %q, got %v", "SF", location)
	}
}

func TestQwen35ParserFakeoutPartialToolCallThenThinkCloseAcrossChunks(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Need weather lookup<tool_c", false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in first chunk, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("</thi", false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "<tool_c" {
		t.Fatalf("expected thinking %q, got %q", "<tool_c", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in second chunk, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("nk>", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking in third chunk, got %q", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in third chunk, got %d", len(calls))
	}
}

func TestQwen35ParserToolCallAfterThinkingCloseIsParsed(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	input := `Need weather lookup</think><tool_call><function=get_weather><parameter=location>
SF
</parameter></function></tool_call>`
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call after </think>, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "SF" {
		t.Fatalf("expected location %q, got %v", "SF", location)
	}
}

func TestQwen35ParserThinkingDisabledPassesContentThrough(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := parser.Add("Plain answer without think close tag.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "Plain answer without think close tag." {
		t.Fatalf("expected content %q, got %q", "Plain answer without think close tag.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserThinkingDisabledWithCloseTagTreatsAsContent(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := parser.Add("</think>Some content after spurious tag.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "</think>Some content after spurious tag." {
		t.Fatalf("expected content %q, got %q", "</think>Some content after spurious tag.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserLeadingThinkCloseProducesContent(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("</think>The final answer.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "The final answer." {
		t.Fatalf("expected content %q, got %q", "The final answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserStreamingSplitThinkCloseTag(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("Reasoning text</thi", false)
	if err != nil {
		t.Fatalf("parse failed on first chunk: %v", err)
	}
	if thinking != "Reasoning text" {
		t.Fatalf("expected thinking %q, got %q", "Reasoning text", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("nk>The final answer.", true)
	if err != nil {
		t.Fatalf("parse failed on second chunk: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking on second chunk, got %q", thinking)
	}
	if content != "The final answer." {
		t.Fatalf("expected content %q, got %q", "The final answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserStreamingEatsWhitespaceAfterThinkClose(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("Reasoning</think>", false)
	if err != nil {
		t.Fatalf("parse failed on first chunk: %v", err)
	}
	if thinking != "Reasoning" {
		t.Fatalf("expected thinking %q, got %q", "Reasoning", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("\n \t", false)
	if err != nil {
		t.Fatalf("parse failed on whitespace chunk: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no thinking on whitespace chunk, got %q", thinking)
	}
	if content != "" {
		t.Fatalf("expected whitespace after </think> to be eaten, got content %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("The final answer.", true)
	if err != nil {
		t.Fatalf("parse failed on content chunk: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking, got %q", thinking)
	}
	if content != "The final answer." {
		t.Fatalf("expected content %q, got %q", "The final answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserDoneDuringWhitespaceEating(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Reasoning</think>", false)
	if err != nil {
		t.Fatalf("parse failed on first chunk: %v", err)
	}
	if thinking != "Reasoning" {
		t.Fatalf("expected thinking %q, got %q", "Reasoning", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}

	content, thinking, calls, err = parser.Add("   ", true)
	if err != nil {
		t.Fatalf("parse failed on trailing whitespace chunk: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking, got %q", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content after whitespace-only done chunk, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserThinkingTruncatedWithoutCloseTag(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Reasoning that never closes", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "Reasoning that never closes" {
		t.Fatalf("expected thinking %q, got %q", "Reasoning that never closes", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}
