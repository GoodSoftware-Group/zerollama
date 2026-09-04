package api

// FilterParallelToolCalls keeps at most one call when the client set
// parallel_tool_calls=false (OpenAI) or disable_parallel_tool_use (Anthropic).
// Nil parallel means the OpenAI default (allow many). already tracks streaming
// so a second parser chunk cannot sneak another call.
func FilterParallelToolCalls(calls []ToolCall, parallel *bool, already *int) []ToolCall {
	if parallel == nil || *parallel || len(calls) == 0 {
		return calls
	}
	n := 0
	if already != nil {
		n = *already
	}
	if n >= 1 {
		return nil
	}
	if already != nil {
		*already = 1
	}
	return calls[:1]
}

// FilterToolsByName keeps tools whose function name matches (OpenAI named
// tool_choice / Anthropic tool_choice.type=tool).
func FilterToolsByName(tools Tools, name string) Tools {
	if name == "" || len(tools) == 0 {
		return nil
	}
	var out Tools
	for _, t := range tools {
		if t.Function.Name == name {
			out = append(out, t)
		}
	}
	return out
}
