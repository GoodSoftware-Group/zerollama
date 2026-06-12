package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/runtimeclient"
)

var runtimeProxyClient = func() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = 120 * time.Second
	return &http.Client{Transport: t}
}()

// runtimeGenerateProxy forwards /api/generate to the Python runtime when resolveRuntimeProxy is true.
// Pulled models send options.gguf from the manifest; X-Zerollama-Runtime: 1 still opts in without a library entry.
func (s *Server) runtimeGenerateProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost || c.Request.URL.Path != "/api/generate" {
			c.Next()
			return
		}

		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		var req api.GenerateRequest
		if err := json.Unmarshal(body, &req); err != nil {
			c.Next()
			return
		}
		if req.Prompt == "" && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
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
	if ollamaWantsStream(req.Stream) {
		payload := map[string]any{
			"model":   req.Model,
			"prompt":  req.Prompt,
			"stream":  true,
			"options": rtOpts,
		}
		if err := writeRuntimeStreamAccepted(c, req.Model, false); err != nil {
			writeRuntimeProxyError(c, err)
			return
		}
		if err := forwardRuntimeNDJSON(c, "/api/generate", payload); err != nil {
				writeRuntimeProxyError(c, err)
			}
			return
		}

		respBody, status, err := forwardRuntimeGenerate(c, req.Model, req.Prompt, rtOpts)
		if err != nil {
			writeRuntimeProxyError(c, err)
			return
		}
		if status >= 300 {
			c.Data(status, "application/json", respBody)
			c.Abort()
			return
		}
		c.Data(http.StatusOK, "application/json", respBody)
		c.Abort()
	}
}

// runtimeProxyEnabled reports whether the sidecar URL is configured (for logs).
func runtimeProxyEnabled() bool {
	return runtimeProxyConfigured()
}
