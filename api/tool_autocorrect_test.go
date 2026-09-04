package api

import "testing"

func testEditFileTool() Tool {
	props := NewToolPropertiesMap()
	props.Set("path", ToolProperty{Type: PropertyType{"string"}})
	editProps := NewToolPropertiesMap()
	editProps.Set("old", ToolProperty{Type: PropertyType{"string"}})
	editProps.Set("new", ToolProperty{Type: PropertyType{"string"}})
	props.Set("edit", ToolProperty{Type: PropertyType{"object"}, Properties: editProps})
	return Tool{Type: "function", Function: ToolFunction{
		Name: "edit_file",
		Parameters: ToolFunctionParameters{
			Type:       "object",
			Required:   []string{"path"},
			Properties: props,
		},
	}}
}

func TestCoerceToolCallsTypes(t *testing.T) {
	props := NewToolPropertiesMap()
	props.Set("n", ToolProperty{Type: PropertyType{"integer"}})
	props.Set("ok", ToolProperty{Type: PropertyType{"boolean"}})
	props.Set("ratio", ToolProperty{Type: PropertyType{"number"}})
	tool := Tool{Type: "function", Function: ToolFunction{
		Name:       "f",
		Parameters: ToolFunctionParameters{Type: "object", Properties: props},
	}}
	args := NewToolCallFunctionArguments()
	args.Set("n", "3")
	args.Set("ok", "true")
	args.Set("ratio", "1.5")
	got := CoerceToolCalls([]ToolCall{{Function: ToolCallFunction{Name: "f", Arguments: args}}}, Tools{tool})
	n, _ := got[0].Function.Arguments.Get("n")
	if n != 3 {
		t.Fatalf("n=%v (%T)", n, n)
	}
	ok, _ := got[0].Function.Arguments.Get("ok")
	if ok != true {
		t.Fatalf("ok=%v", ok)
	}
	ratio, _ := got[0].Function.Arguments.Get("ratio")
	if ratio != 1.5 {
		t.Fatalf("ratio=%v", ratio)
	}
}

func TestCoerceToolCallsHoistBuriedPath(t *testing.T) {
	edit := map[string]any{"path": "/tmp/a.go", "old": "x", "new": "y"}
	args := NewToolCallFunctionArguments()
	args.Set("edit", edit)
	got := CoerceToolCalls([]ToolCall{{Function: ToolCallFunction{Name: "edit_file", Arguments: args}}}, Tools{testEditFileTool()})
	path, ok := got[0].Function.Arguments.Get("path")
	if !ok || path != "/tmp/a.go" {
		t.Fatalf("path=%v ok=%v", path, ok)
	}
	nested, _ := got[0].Function.Arguments.Get("edit")
	m := nested.(map[string]any)
	if _, still := m["path"]; still {
		t.Fatalf("path still nested: %v", m)
	}
	if m["old"] != "x" || m["new"] != "y" {
		t.Fatalf("edit=%v", m)
	}
}

func TestCoerceToolCallsLeavesCorrectCall(t *testing.T) {
	args := NewToolCallFunctionArguments()
	args.Set("path", "ok")
	args.Set("edit", map[string]any{"old": "a", "new": "b"})
	got := CoerceToolCalls([]ToolCall{{Function: ToolCallFunction{Name: "edit_file", Arguments: args}}}, Tools{testEditFileTool()})
	path, _ := got[0].Function.Arguments.Get("path")
	if path != "ok" {
		t.Fatalf("path=%v", path)
	}
}
