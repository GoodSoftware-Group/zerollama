package server

import (
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// applyThinkingGate enforces ZEROLLAMA_THINKING_GATE on a request that would
// think (minefield trap 29 — server off is otherwise only a default).
// Returns a client-facing error when mode is deny; otherwise mutates think.
func applyThinkingGate(think **api.ThinkValue) error {
	if think == nil || *think == nil || !(*think).Bool() {
		return nil
	}
	switch envconfig.ThinkingGate() {
	case "deny":
		return fmt.Errorf("thinking disabled by ZEROLLAMA_THINKING_GATE=deny")
	case "strip":
		off := &api.ThinkValue{Value: false}
		*think = off
		return nil
	default:
		return nil
	}
}
