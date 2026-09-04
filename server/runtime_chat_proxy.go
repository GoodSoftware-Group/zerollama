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
// WHY applyRequestTimeout here: this middleware aborts before ChatHandler when it proxies;
// without the wrap, timeout only worked on the legacy ggml fallthrough path (M15e audit).
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
		// Trap 77: reject unknown top-level fields before runtime forward.
		if err := api.CheckUnknownChatFields(body); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.Request.Body = io.NopCloser(strings.NewReader(string(body)))

		var req api.ChatRequest
		if err := json.Unmarshal(body, &req); err != nil {
			c.Next()
			return
		}
		EnsureAgentPromptCacheKey(&req)
		if err := api.ApplyChatThinkingAliases(&req); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if len(req.Tools) > 0 && formatHasGrammarConstraint(req.Format) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errGrammarWithTools})
			return
		}

		reqCtx, cancelTimeout := applyRequestTimeout(c.Request.Context(), req.Timeout)
		if cancelTimeout != nil {
			defer cancelTimeout()
		}
		c.Request = c.Request.WithContext(reqCtx)
		if req.Timeout != nil {
			c.Set("request_timeout", req.Timeout)
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
		if !resolveRuntimeProxy(c, req.Model, req.Options) {
			c.Next()
			return
		}
		if s.abortIfTrainingBusy(c) {
			return
		}

		nPredict, limited := numPredictFromOptions(req.Options)
		rtOpts := runtimeProxyOptions(req.Model, nPredict, limited, req.Options)
		if limited {
			req.Messages = api.AppendOutputBudgetGuidance(req.Messages, nPredict)
		}
		gguf, _ := rtOpts["gguf"].(string)
		if s.abortIfPrepareRuntimeVRAMFailed(c, s.prepareRuntimeVRAM(c.Request.Context(), gguf, runtimeForceUnload(s, req.Options))) {
			return
		}
		if gguf != "" {
			runtimeclient.LogVramBudgetIfTight(c.Request.Context(), req.Model, gguf, rtOpts)
		}

		stream := ollamaWantsStream(req.Stream)
		nctx := numCtxFromChatOptions(rtOpts)
		if nctx <= 0 {
			nctx = numCtxFromChatOptions(req.Options)
		}
		msgs, compressionMeta, cerr := applyChatCompressionForRequest(c.Request.Context(), &req, req.Messages, nctx, req.Model, 0, s.summarizeChatHead)
		if isChatCompressOverflow(cerr) {
			c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{"error": cerr.Error()})
			return
		}
		if cerr != nil {
			writeRuntimeProxyError(c, cerr)
			return
		}
		req.Messages = msgs
		payload := runtimeChatPayload(req, rtOpts, stream)
		if stream {
			if err := writeRuntimeStreamAccepted(c, req.Model, true); err != nil {
				writeRuntimeProxyError(c, err)
				return
			}
		}
		if stream {
			if err := forwardRuntimeNDJSON(c, "/api/chat", payload, compressionMeta); err != nil {
				writeRuntimeProxyError(c, err)
			}
			return
		}
		if err := forwardRuntimeChatJSON(c, payload, compressionMeta); err != nil {
			writeRuntimeProxyError(c, err)
		}
	}
}
