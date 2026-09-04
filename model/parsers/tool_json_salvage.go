package parsers

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/ollama/ollama/api"
)

var errEmptyToolName = errors.New("empty function name")

// salvageJSONToolCall recovers a function name from mangled or truncated
// tool-call JSON and ships empty arguments (mlx-serve: never partial values).
func salvageJSONToolCall(raw string) (api.ToolCall, bool) {
	name := jsonStringField(raw, "name")
	if name == "" || strings.Contains(name, "<|") {
		return api.ToolCall{}, false
	}
	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      name,
			Arguments: api.NewToolCallFunctionArguments(),
		},
	}, true
}

func jsonStringField(raw, key string) string {
	needle := `"` + key + `"`
	i := 0
	for i < len(raw) {
		j := strings.Index(raw[i:], needle)
		if j < 0 {
			return ""
		}
		j += i
		rest := strings.TrimLeft(raw[j+len(needle):], " \t\n\r")
		if !strings.HasPrefix(rest, ":") {
			i = j + 1
			continue
		}
		rest = strings.TrimLeft(rest[1:], " \t\n\r")
		if rest == "" || rest[0] != '"' {
			i = j + 1
			continue
		}
		var s string
		if json.NewDecoder(strings.NewReader(rest)).Decode(&s) != nil {
			return ""
		}
		return s
	}
	return ""
}

// salvageGLM46ToolCall recovers the function name from an unclosed GLM
// <tool_call> body and ships empty arguments (never partial arg_value text).
func salvageGLM46ToolCall(raw string) (api.ToolCall, bool) {
	name := glm46ToolName(raw)
	if name == "" || strings.Contains(name, "<") {
		return api.ToolCall{}, false
	}
	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      name,
			Arguments: api.NewToolCallFunctionArguments(),
		},
	}, true
}

func glm46ToolName(raw string) string {
	s := strings.TrimSpace(raw)
	s = strings.TrimPrefix(s, glm46ToolOpenTag)
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '<'); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if i := strings.IndexAny(s, "\n\r\t "); i >= 0 {
		s = s[:i]
	}
	return s
}
