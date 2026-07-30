package openai

import (
	"encoding/json"
	"strings"
)

// formatHasGrammar reports whether format constrains decode (json / schema / gbnf).
// WHY: tools + grammar is ambiguous — reject loud (M15f) instead of silent override.
func formatHasGrammar(format json.RawMessage) bool {
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
	// JSON Schema object
	return true
}
