package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/runtimeclient"
)

// runtimeChatProxy forwards /api/chat to the Python runtime when eligible (streaming or not).
func (s *Server) runtimeChatProxy() gin.HandlerFunc {
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
		c.Request.Body = io.NopCloser(strings.NewReader(string(body)))

		var req api.ChatRequest
		if err := json.Unmarshal(body, &req); err != nil {
			c.Next()
			return
		}
		if len(req.Messages) == 0 && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
			c.Next()
			return
		}
		if chatNeedsLegacyRunner(req.Messages, req) {
			c.Next()
			return
		}

		modelRef, err := parseAndValidateModelRef(req.Model)
		if err != nil {
			c.Next()
			return
		}
		if modelRef.Source == modelSourceCloud {
			c.Next()
			return
		}
		if !resolveRuntimeProxy(c, req.Model) {
			c.Next()
			return
		}
		if s.abortIfTrainingBusy(c) {
			return
		}

		s.prepareRuntimeVRAM(c.Request.Context())

		nPredict, limited := numPredictFromOptions(req.Options)
		rtOpts := runtimeProxyOptions(req.Model, nPredict, limited, req.Options)
		if gguf, ok := rtOpts["gguf"].(string); ok {
			runtimeclient.LogVramBudgetIfTight(c.Request.Context(), req.Model, gguf, rtOpts)
		}

		stream := ollamaWantsStream(req.Stream)
		payload := runtimeChatPayload(req, rtOpts, stream)
		if stream {
			if err := forwardRuntimeNDJSON(c, "/api/chat", payload); err != nil {
				writeRuntimeProxyError(c, err)
			}
			return
		}
		if err := forwardRuntimeChatJSON(c, payload); err != nil {
			writeRuntimeProxyError(c, err)
		}
	}
}
