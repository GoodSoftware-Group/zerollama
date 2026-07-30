package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/internal/runtimeclient"
	"github.com/ollama/ollama/openai"
)

// runtimeV1ChatCompletionsProxy forwards text-only OpenAI chat requests to the Python runtime.
// Injects options.gguf from the manifest (Phase 9) so Python Phase 13 VRAM policy can resolve num_ctx.
func (s *Server) runtimeV1ChatCompletionsProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost || c.Request.URL.Path != "/v1/chat/completions" {
			c.Next()
			return
		}

		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		// Trap 77: reject unknown top-level fields before runtime forward (same floor as ChatMiddleware).
		if err := openai.CheckUnknownChatCompletionFields(body); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, openai.NewError(http.StatusBadRequest, err.Error()))
			return
		}

		var oreq openai.ChatCompletionRequest
		if err := json.Unmarshal(body, &oreq); err != nil {
			c.Next()
			return
		}

		reqCtx, cancelTimeout := applyRequestTimeout(c.Request.Context(), oreq.Timeout)
		if cancelTimeout != nil {
			defer cancelTimeout()
		}
		c.Request = c.Request.WithContext(reqCtx)
		if oreq.Timeout != nil {
			c.Set("request_timeout", oreq.Timeout)
		}

		var bodyMap map[string]any
		if err := json.Unmarshal(body, &bodyMap); err != nil {
			c.Next()
			return
		}
		if v1ChatNeedsLegacyRunner(&oreq, bodyMap) {
			c.Next()
			return
		}

		modelRef, err := parseAndValidateModelRef(oreq.Model)
		if err != nil {
			c.Next()
			return
		}
		if modelRef.Source == modelSourceCloud {
			c.Next()
			return
		}
		if !resolveRuntimeProxy(c, oreq.Model, proxyOptsFromV1Body(bodyMap)) {
			c.Next()
			return
		}
		if s.abortIfTrainingBusy(c) {
			return
		}

		rtOpts := runtimeV1ProxyOptions(oreq.Model, bodyMap)
		gguf, _ := rtOpts["gguf"].(string)
		if s.abortIfPrepareRuntimeVRAMFailed(c, s.prepareRuntimeVRAM(c.Request.Context(), gguf, runtimeForceUnload(s, proxyOptsFromV1Body(bodyMap)))) {
			return
		}
		bodyMap["options"] = rtOpts
		proxyBody, err := json.Marshal(bodyMap)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if gguf != "" && strings.TrimSpace(gguf) != "" {
			runtimeclient.LogVramBudgetIfTight(c.Request.Context(), oreq.Model, gguf, rtOpts)
		}

		target := effectiveRuntimeURL() + "/v1/chat/completions"
		outReq, err := http.NewRequestWithContext(
			c.Request.Context(),
			http.MethodPost,
			target,
			bytes.NewReader(proxyBody),
		)
		if err != nil {
			slog.Error("runtime v1 proxy: build request", "error", err)
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		ct := c.GetHeader("Content-Type")
		if ct == "" {
			ct = "application/json"
		}
		outReq.Header.Set("Content-Type", ct)
		if accept := c.GetHeader("Accept"); accept != "" {
			outReq.Header.Set("Accept", accept)
		}
		if auth := c.GetHeader("Authorization"); auth != "" {
			outReq.Header.Set("Authorization", auth)
		}

		resp, err := runtimeProxyClient.Do(outReq)
		if err != nil {
			writeRuntimeProxyError(c, err)
			return
		}
		defer resp.Body.Close()

		for k, vals := range resp.Header {
			if strings.EqualFold(k, "Content-Length") {
				continue
			}
			for _, v := range vals {
				c.Writer.Header().Add(k, v)
			}
		}
		c.Status(resp.StatusCode)
		if err := copyRuntimeResponseBody(c.Writer, resp.Body); err != nil {
			slog.Warn("runtime v1 proxy: copy body", "error", err)
		}
		c.Abort()
	}
}

func v1ChatNeedsLegacyRunner(oreq *openai.ChatCompletionRequest, body map[string]any) bool {
	if oreq == nil {
		return true
	}
	if v1BodyNeedsLegacyRunner(body) {
		return true
	}
	if oreq.Logprobs != nil && *oreq.Logprobs {
		return true
	}
	if oreq.Reasoning != nil {
		return true
	}
	if oreq.ReasoningEffort != nil && strings.TrimSpace(*oreq.ReasoningEffort) != "" {
		return true
	}
	if openai.ChatCompletionRequestHasVideoURL(oreq) {
		return true
	}
	for _, msg := range oreq.Messages {
		if strings.TrimSpace(msg.Reasoning) != "" {
			return true
		}
		parts, ok := msg.Content.([]any)
		if !ok {
			continue
		}
		for _, p := range parts {
			m, ok := p.(map[string]any)
			if !ok {
				continue
			}
			t, _ := m["type"].(string)
			switch strings.ToLower(t) {
			case "image_url", "input_image":
				return true
			}
		}
	}
	return false
}

// v1ThinkNeedsLegacy mirrors Python _v1_think_needs_legacy (think:false stays on runtime).
func v1ThinkNeedsLegacy(v any) bool {
	if v == nil {
		return false
	}
	if b, ok := v.(bool); ok {
		return b
	}
	if s, ok := v.(string); ok {
		return strings.TrimSpace(s) != ""
	}
	return true
}

// v1BodyNeedsLegacyRunner inspects raw JSON fields not mapped on ChatCompletionRequest.
func v1BodyNeedsLegacyRunner(body map[string]any) bool {
	if body == nil {
		return false
	}
	if v1ThinkNeedsLegacy(body["think"]) {
		return true
	}
	if body["reasoning"] != nil {
		return true
	}
	if re, ok := body["reasoning_effort"].(string); ok && strings.TrimSpace(re) != "" {
		return true
	}
	return false
}
