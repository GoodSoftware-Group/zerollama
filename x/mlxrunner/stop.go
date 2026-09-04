package mlxrunner

import (
	"strings"

	"github.com/ollama/ollama/runner/common"
)

func nonemptyStops(stops []string) []string {
	out := make([]string, 0, len(stops))
	for _, s := range stops {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// flushStopHold holds generated text that could still be a stop-sequence prefix
// (same as ollamarunner / llama-server). force emits the hold even if it is a
// prefix (end of request). hit means a full stop was found and stripped.
func flushStopHold(pending string, stops []string, force bool) (emit, remainder, matched string) {
	if pending == "" {
		return "", "", ""
	}
	if len(stops) == 0 {
		return pending, "", ""
	}
	if ok, stop := common.FindStop(pending, stops); ok {
		idx := strings.Index(pending, stop)
		return pending[:idx], "", stop
	}
	if !force && common.ContainsStopSuffix(pending, stops) {
		return "", pending, ""
	}
	return pending, "", ""
}
