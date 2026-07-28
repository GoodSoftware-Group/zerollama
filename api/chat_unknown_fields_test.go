package api

import (
	"strings"
	"testing"
)

func TestCheckUnknownChatFields_RejectsMinefieldProbe(t *testing.T) {
	err := CheckUnknownChatFields([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"__minefield_unvalidated_field_probe__": true
	}`))
	if err == nil {
		t.Fatal("expected unknown field error")
	}
	if !strings.Contains(err.Error(), "__minefield_unvalidated_field_probe__") {
		t.Fatalf("error = %v", err)
	}
}

func TestCheckUnknownChatFields_AllowsKnown(t *testing.T) {
	err := CheckUnknownChatFields([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"stream": false,
		"think": false,
		"options": {"temperature": 0},
		"keep_alive": "5m"
	}`))
	if err != nil {
		t.Fatal(err)
	}
}
