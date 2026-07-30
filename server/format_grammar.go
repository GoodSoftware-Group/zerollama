package server

import (
	"encoding/json"
	"strings"
)

// errGrammarWithTools is the loud 400 when tools and format both constrain decode (M15f).
const errGrammarWithTools = "cannot use tools with format (json/schema/gbnf); drop one"

// formatHasGrammarConstraint reports whether format constrains decode.
// Mirrored from openai.formatHasGrammar so /api/chat can reject tools+format.
func formatHasGrammarConstraint(format json.RawMessage) bool {
	if len(format) == 0 {
		return false
	}
	switch string(format) {
	case `null`, `""`:
		return false
	case `"json"`:
		return true
	}
	if format[0] != '{' {
		return true
	}
	var probe struct {
		Type    string `json:"type"`
		Grammar string `json:"grammar"`
	}
	if err := json.Unmarshal(format, &probe); err != nil {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(probe.Type), "gbnf") {
		return strings.TrimSpace(probe.Grammar) != ""
	}
	return true
}
