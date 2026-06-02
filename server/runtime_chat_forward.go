package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

func chatMessagesToMaps(messages []api.Message) []map[string]any {
	out := make([]map[string]any, 0, len(messages))
	for _, m := range messages {
		msg := map[string]any{
			"role":    m.Role,
			"content": m.Content,
		}
		if len(m.ToolCalls) > 0 {
			msg["tool_calls"] = m.ToolCalls
		}
		if m.ToolName != "" {
			msg["tool_name"] = m.ToolName
		}
		if m.ToolCallID != "" {
			msg["tool_call_id"] = m.ToolCallID
		}
		out = append(out, msg)
	}
	return out
}

func toolsToMaps(tools api.Tools) []map[string]any {
	if len(tools) == 0 {
		return nil
	}
	raw, err := json.Marshal(tools)
	if err != nil {
		return nil
	}
	var out []map[string]any
	if json.Unmarshal(raw, &out) != nil {
		return nil
	}
	return out
}

func runtimeChatPayload(
	req api.ChatRequest,
	rtOpts map[string]any,
	stream bool,
) map[string]any {
	payload := map[string]any{
		"model":    req.Model,
		"messages": chatMessagesToMaps(req.Messages),
		"stream":   stream,
		"options":  rtOpts,
	}
	if tools := toolsToMaps(req.Tools); len(tools) > 0 {
		payload["tools"] = tools
	}
	if req.Logprobs {
		payload["logprobs"] = true
	}
	if req.Think != nil {
		payload["think"] = req.Think
	}
	return payload
}

func forwardRuntimeChatJSON(
	c *gin.Context,
	payload map[string]any,
) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	target := effectiveRuntimeURL() + "/api/chat"
	outReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		http.MethodPost,
		target,
		bytes.NewReader(body),
	)
	if err != nil {
		return err
	}
	outReq.Header.Set("Content-Type", "application/json")

	resp, err := runtimeProxyClient.Do(outReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		c.Data(resp.StatusCode, "application/json", respBody)
		c.Abort()
		return nil
	}
	c.Data(http.StatusOK, "application/json", respBody)
	c.Abort()
	return nil
}
