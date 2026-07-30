package server

import (
	"encoding/json"
	"testing"
)

func TestFormatHasGrammarConstraint(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		raw  string
		want bool
	}{
		{"empty", "", false},
		{"null", "null", false},
		{"empty_string", `""`, false},
		{"json_mode", `"json"`, true},
		{"schema", `{"type":"object"}`, true},
		{"gbnf_ok", `{"type":"gbnf","grammar":"root ::= \"x\""}`, true},
		{"gbnf_empty", `{"type":"gbnf","grammar":""}`, false},
		{"gbnf_case", `{"type":"GBNF","grammar":"root ::= \"x\""}`, true},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			var raw json.RawMessage
			if tt.raw != "" {
				raw = json.RawMessage(tt.raw)
			}
			if got := formatHasGrammarConstraint(raw); got != tt.want {
				t.Fatalf("got %v want %v for %s", got, tt.want, tt.raw)
			}
		})
	}
}
