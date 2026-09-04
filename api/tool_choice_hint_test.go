package api

import "testing"

func TestAppendRequiredToolCallHint(t *testing.T) {
	tools := []Tool{{Function: ToolFunction{Name: "t"}}}
	got := AppendRequiredToolCallHint([]Message{{Role: "user", Content: "hi"}}, tools, true)
	if got[0].Content != "hi\n\n"+RequiredToolCallHint {
		t.Fatalf("got %q", got[0].Content)
	}
	again := AppendRequiredToolCallHint(got, tools, true)
	if again[0].Content != got[0].Content {
		t.Fatal("not idempotent")
	}
	if AppendRequiredToolCallHint([]Message{{Role: "user", Content: "hi"}}, tools, false)[0].Content != "hi" {
		t.Fatal("auto must not hint")
	}
	if AppendRequiredToolCallHint([]Message{{Role: "user", Content: "hi"}}, nil, true)[0].Content != "hi" {
		t.Fatal("no tools must not hint")
	}
}

func TestToolChoiceRequiresCall(t *testing.T) {
	if !ToolChoiceRequiresCall("required") || !ToolChoiceRequiresCall("ANY") {
		t.Fatal("required/any")
	}
	if ToolChoiceRequiresCall("auto") || ToolChoiceRequiresCall("none") {
		t.Fatal("auto/none")
	}
}
