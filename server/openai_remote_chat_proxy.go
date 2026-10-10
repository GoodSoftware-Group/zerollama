package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

// openaiRemoteProxyClient bounds TTFB; SSE streams still use the body copy.
var openaiRemoteProxyClient = func() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = 120 * time.Second
	return &http.Client{Transport: t}
}()

// openaiRemoteBaseURL returns per-model backend_paths.openai_url, else fleet env.
func openaiRemoteBaseURL(m *Model) string {
	if m != nil && m.Config.BackendPaths != nil {
		if u := strings.TrimSpace(m.Config.BackendPaths["openai_url"]); u != "" {
			return strings.TrimSuffix(u, "/")
		}
	}
	return envconfig.OpenAIRemoteURL()
}

// openaiRemoteUpstreamModel is the model id the sidecar expects.
func openaiRemoteUpstreamModel(m *Model, reqModel string) string {
	if m != nil && m.Config.BackendPaths != nil {
		if u := strings.TrimSpace(m.Config.BackendPaths["openai_model"]); u != "" {
			return u
		}
	}
	name := model.ParseName(reqModel)
	if name.IsValid() {
		// Sidecars usually register short ids (glm-5.3-flash), not host/namespace:tag.
		if name.Model != "" {
			return name.Model
		}
	}
	return strings.TrimSuffix(reqModel, ":latest")
}

func openaiRemoteShouldProxy(m *Model) bool {
	return m != nil && modelInferenceBackend(m) == model.BackendOpenAIRemote
}

func openaiRemoteLookupModel(modelName string) (*Model, bool) {
	modelRef, err := parseAndValidateModelRef(modelName)
	if err != nil || modelRef.Source == modelSourceCloud {
		return nil, false
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		return nil, false
	}
	m, err := GetModel(name.String())
	if err != nil || !openaiRemoteShouldProxy(m) {
		return nil, false
	}
	return m, true
}

func openaiRemoteForwardRaw(c *gin.Context, base string, body []byte) {
	target := base + "/v1/chat/completions"
	outReq, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		slog.Error("openai-remote proxy: build request", "error", err)
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

	resp, err := openaiRemoteProxyClient.Do(outReq)
	if err != nil {
		slog.Warn("openai-remote proxy: request failed", "error", err)
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
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
	if _, err := io.Copy(c.Writer, resp.Body); err != nil {
		slog.Debug("openai-remote proxy: copy response", "error", err)
	}
	c.Abort()
}

// openaiRemoteChatCompletionsProxy forwards full JSON to an OpenAI-compatible sidecar.
// External VRAM — no GetRunner / no GGUF load (GF3 / glm-flash-lite).
func (s *Server) openaiRemoteChatCompletionsProxy() gin.HandlerFunc {
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

		var oreq openai.ChatCompletionRequest
		if err := json.Unmarshal(body, &oreq); err != nil {
			c.Next()
			return
		}

		m, ok := openaiRemoteLookupModel(oreq.Model)
		if !ok {
			c.Next()
			return
		}

		base := openaiRemoteBaseURL(m)
		if base == "" {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
				"error": "modality_backends.inference=openai-remote requires backend_paths.openai_url or ZEROLLAMA_OPENAI_REMOTE_URL",
			})
			return
		}

		upstreamModel := openaiRemoteUpstreamModel(m, oreq.Model)
		if upstreamModel != oreq.Model {
			oreq.Model = upstreamModel
			body, err = json.Marshal(oreq)
			if err != nil {
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
		}

		slog.Debug("openai-remote proxy: forwarding v1 chat",
			"model", oreq.Model,
			"upstream_model", upstreamModel,
			"target", base+"/v1/chat/completions",
		)
		openaiRemoteForwardRaw(c, base, body)
	}
}

// openaiRemoteOllamaChatRequest builds a text-only OpenAI chat body from /api/chat.
func openaiRemoteOllamaChatRequest(m *Model, req api.ChatRequest) (openai.ChatCompletionRequest, error) {
	msgs := make([]openai.Message, 0, len(req.Messages))
	for _, msg := range req.Messages {
		if len(msg.Images) > 0 || len(msg.Videos) > 0 || len(msg.AudioClips) > 0 {
			return openai.ChatCompletionRequest{}, fmt.Errorf("openai-remote does not support multimodal /api/chat (use text-only or /v1)")
		}
		if len(msg.ToolCalls) > 0 {
			return openai.ChatCompletionRequest{}, fmt.Errorf("openai-remote does not support tools on /api/chat yet (use /v1)")
		}
		msgs = append(msgs, openai.Message{Role: msg.Role, Content: msg.Content})
	}
	oreq := openai.ChatCompletionRequest{
		Model:              openaiRemoteUpstreamModel(m, req.Model),
		Messages:           msgs,
		Stream:             false,
		ChatTemplateKwargs: map[string]any{"enable_thinking": false},
	}
	if n := api.NumPredictFromMap(req.Options); n > 0 {
		oreq.MaxTokens = &n
	}
	return oreq, nil
}

func openaiRemoteToOllamaChat(modelName string, oc openai.ChatCompletion) api.ChatResponse {
	content := ""
	role := "assistant"
	doneReason := "stop"
	if len(oc.Choices) > 0 {
		ch := oc.Choices[0]
		role = ch.Message.Role
		if role == "" {
			role = "assistant"
		}
		switch v := ch.Message.Content.(type) {
		case string:
			content = v
		case nil:
		default:
			b, _ := json.Marshal(v)
			content = string(b)
		}
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			doneReason = *ch.FinishReason
		}
	}
	return api.ChatResponse{
		Model:     modelName,
		CreatedAt: time.Now(),
		Message:   api.Message{Role: role, Content: content},
		Done:      true,
		DoneReason: doneReason,
		Metrics: api.Metrics{
			PromptEvalCount: oc.Usage.PromptTokens,
			EvalCount:       oc.Usage.CompletionTokens,
		},
	}
}

// openaiRemoteChatProxy translates /api/chat → sidecar OpenAI /v1 (non-stream upstream).
// Stream clients get a single NDJSON terminal chunk (CLI-friendly).
func (s *Server) openaiRemoteChatProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost || c.Request.URL.Path != "/api/chat" {
			c.Next()
			return
		}

		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		var req api.ChatRequest
		if err := json.Unmarshal(body, &req); err != nil {
			c.Next()
			return
		}

		m, ok := openaiRemoteLookupModel(req.Model)
		if !ok {
			c.Next()
			return
		}

		base := openaiRemoteBaseURL(m)
		if base == "" {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
				"error": "modality_backends.inference=openai-remote requires backend_paths.openai_url or ZEROLLAMA_OPENAI_REMOTE_URL",
			})
			return
		}

		oreq, err := openaiRemoteOllamaChatRequest(m, req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		payload, err := json.Marshal(oreq)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		target := base + "/v1/chat/completions"
		outReq, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, target, bytes.NewReader(payload))
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		outReq.Header.Set("Content-Type", "application/json")

		slog.Debug("openai-remote proxy: forwarding /api/chat",
			"model", req.Model,
			"upstream_model", oreq.Model,
			"target", target,
		)

		resp, err := openaiRemoteProxyClient.Do(outReq)
		if err != nil {
			slog.Warn("openai-remote /api/chat: request failed", "error", err)
			c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		defer resp.Body.Close()
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		if resp.StatusCode >= 400 {
			c.Data(resp.StatusCode, "application/json", respBody)
			c.Abort()
			return
		}

		var oc openai.ChatCompletion
		if err := json.Unmarshal(respBody, &oc); err != nil {
			c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "openai-remote: bad upstream JSON: " + err.Error()})
			return
		}
		out := openaiRemoteToOllamaChat(req.Model, oc)

		stream := req.Stream == nil || *req.Stream
		if stream {
			c.Header("Content-Type", "application/x-ndjson")
			c.Status(http.StatusOK)
			enc := json.NewEncoder(c.Writer)
			if err := enc.Encode(out); err != nil {
				slog.Debug("openai-remote /api/chat: encode", "error", err)
			}
			c.Abort()
			return
		}
		c.JSON(http.StatusOK, out)
		c.Abort()
	}
}
