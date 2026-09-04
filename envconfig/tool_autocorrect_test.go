package envconfig

import "testing"

func TestToolAutocorrectDefaultOn(t *testing.T) {
	t.Setenv("ZEROLLAMA_TOOL_AUTOCORRECT", "")
	if !ToolAutocorrect() {
		t.Fatal("default on")
	}
	t.Setenv("ZEROLLAMA_TOOL_AUTOCORRECT", "0")
	if ToolAutocorrect() {
		t.Fatal("0 disables")
	}
}
