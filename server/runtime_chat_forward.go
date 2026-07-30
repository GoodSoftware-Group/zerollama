package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"

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
	// M15f: forward format so Python llama-server can apply json_schema/grammar.
	// WHY: previously dropped → unconstrained output on runtime-proxied models.
	if len(req.Format) > 0 {
		payload["format"] = json.RawMessage(req.Format)
	}
	return payload
}

func recordRuntimeProxyErrorMetrics(status int, respBody []byte) {
	if status == http.StatusServiceUnavailable {
		metricsIncQueueReject()
	}
	errText := string(respBody)
	var errObj map[string]any
	if json.Unmarshal(respBody, &errObj) == nil {
		if e, ok := errObj["error"].(string); ok {
			errText = e
		}
	}
	if isHostUnstableError(strings.ToLower(errText)) {
		metricsIncRunnerCrash()
		metricsIncRequestResult("host_unstable")
		return
	}
	metricsIncRequestResult("error")
}

func injectRuntimeRetryAfter(c *gin.Context, status int, respBody []byte) []byte {
	if status != http.StatusServiceUnavailable {
		return respBody
	}
	c.Header("Retry-After", strconv.Itoa(defaultBusyRetryAfterSec))
	var obj map[string]any
	if json.Unmarshal(respBody, &obj) == nil {
		if _, has := obj["retry_after"]; !has {
			obj["retry_after"] = defaultBusyRetryAfterSec
		}
		if patched, err := json.Marshal(obj); err == nil {
			return patched
		}
	}
	return respBody
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
		respBody = injectRuntimeRetryAfter(c, resp.StatusCode, respBody)
		recordRuntimeProxyErrorMetrics(resp.StatusCode, respBody)
		c.Data(resp.StatusCode, "application/json", respBody)
		c.Abort()
		return nil
	}
	metricsIncRequestResult("ok")
	c.Data(http.StatusOK, "application/json", respBody)
	c.Abort()
	return nil
}
