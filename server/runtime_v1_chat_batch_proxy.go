package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/internal/runtimeclient"
)

// runtimeV1ChatCompletionsBatchProxy forwards POST /v1/chat/completions/batch to Python.
//
// WHY thin proxy (not sched.go): Go pending_queue is model-load only; Python already
// does decode batching via generate_batch. Same-model + shared options.gguf required.
// WHY reject tools/vision/think: those need the interactive chat path; batch is text fan-out.
// WHY same-model only: one GGUF / one runner — mixed models would need a second scheduler;
// Hermes must group by model client-side (document that, don't invent server grouping).
// WHY wrapper response (not bare []chat.completion): object=chat.completion.batch +
// ordered completions[] is the stable contract — see OpenAPI ChatCompletionsBatchResponse
// and docs/hermes-zerollama-gap.md §8. Underspecified "OpenAI-shaped list" blocked clients.
func (s *Server) runtimeV1ChatCompletionsBatchProxy() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost || c.Request.URL.Path != "/v1/chat/completions/batch" {
			c.Next()
			return
		}

		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		var bodyMap map[string]any
		if err := json.Unmarshal(body, &bodyMap); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid JSON"})
			return
		}
		if stream, ok := bodyMap["stream"].(bool); ok && stream {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
				"error": "batch chat streaming not supported yet",
			})
			return
		}

		reqs, _ := bodyMap["requests"].([]any)
		if len(reqs) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "requests must be a non-empty list"})
			return
		}
		if len(reqs) > 8 {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "batch size exceeds max of 8"})
			return
		}

		model := strings.TrimSpace(asString(bodyMap["model"]))
		for i, raw := range reqs {
			item, ok := raw.(map[string]any)
			if !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "each request must be an object"})
				return
			}
			if v1BodyNeedsLegacyRunner(item) || item["tools"] != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
					"error": "batch chat does not support tools/vision/think",
				})
				return
			}
			itemModel := strings.TrimSpace(asString(item["model"]))
			if itemModel == "" {
				itemModel = model
			}
			if itemModel == "" {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
				return
			}
			if model == "" {
				model = itemModel
			} else if itemModel != model {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
					"error": "batch chat requires the same model for all requests",
				})
				return
			}
			_ = i
		}
		bodyMap["model"] = model

		modelRef, err := parseAndValidateModelRef(model)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
			return
		}
		if modelRef.Source == modelSourceCloud {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errCloudUseOpenAICompat})
			return
		}
		if !resolveRuntimeProxy(c, model, proxyOptsFromV1Body(bodyMap)) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
				"error": "batch chat requires the Python runtime path (set modality_backends.inference or ZEROLLAMA_RUNTIME)",
			})
			return
		}
		if s.abortIfTrainingBusy(c) {
			return
		}

		rtOpts := runtimeV1ProxyOptions(model, bodyMap)
		gguf, _ := rtOpts["gguf"].(string)
		if s.abortIfPrepareRuntimeVRAMFailed(c, s.prepareRuntimeVRAM(c.Request.Context(), gguf, runtimeForceUnload(s, proxyOptsFromV1Body(bodyMap)))) {
			return
		}
		bodyMap["options"] = rtOpts
		// Ensure each nested request inherits shared options.gguf when unset.
		for _, raw := range reqs {
			item, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			opts, _ := item["options"].(map[string]any)
			if opts == nil {
				opts = map[string]any{}
			}
			for k, v := range rtOpts {
				if _, exists := opts[k]; !exists {
					opts[k] = v
				}
			}
			item["options"] = opts
		}
		proxyBody, err := json.Marshal(bodyMap)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if strings.TrimSpace(gguf) != "" {
			runtimeclient.LogVramBudgetIfTight(c.Request.Context(), model, gguf, rtOpts)
		}

		target := effectiveRuntimeURL() + "/v1/chat/completions/batch"
		outReq, err := http.NewRequestWithContext(
			c.Request.Context(),
			http.MethodPost,
			target,
			bytes.NewReader(proxyBody),
		)
		if err != nil {
			writeRuntimeProxyError(c, err)
			return
		}
		outReq.Header.Set("Content-Type", "application/json")
		resp, err := runtimeProxyClient.Do(outReq)
		if err != nil {
			writeRuntimeProxyError(c, err)
			return
		}
		defer resp.Body.Close()
		c.Status(resp.StatusCode)
		for k, vv := range resp.Header {
			if strings.EqualFold(k, "Content-Length") {
				continue
			}
			for _, v := range vv {
				c.Writer.Header().Add(k, v)
			}
		}
		if err := copyRuntimeResponseBody(c.Writer, resp.Body); err != nil {
			writeRuntimeProxyError(c, err)
			return
		}
		c.Abort()
	}
}

func asString(v any) string {
	s, _ := v.(string)
	return s
}
