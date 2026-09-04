package api

import (
	"fmt"
	"strings"
)

// EnsureToolCallID returns a non-empty tool-call id for history / OpenAI wire.
// mlx-serve extra-context: Jinja that does tool_calls[].id or arguments|items
// must see an id on every history call. Empty ids stay empty until here so a
// client-supplied id is never rewritten.
func EnsureToolCallID(id string, index int) string {
	if s := strings.TrimSpace(id); s != "" {
		return s
	}
	if index < 0 {
		index = 0
	}
	return fmt.Sprintf("call_%d", index)
}
