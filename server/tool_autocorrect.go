package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

func finishToolCalls(calls []api.ToolCall, req api.ChatRequest, already *int) []api.ToolCall {
	calls = api.FilterParallelToolCalls(calls, req.ParallelToolCalls, already)
	if envconfig.ToolAutocorrect() {
		calls = api.CoerceToolCalls(calls, req.Tools)
	}
	return calls
}
