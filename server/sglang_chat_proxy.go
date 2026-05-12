package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

// sglangProxyClient uses a bounded time-to-first-byte for the upstream while still allowing long-lived SSE streams.
var sglangProxyClient = func() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = 60 * time.Second
	return &http.Client{Transport: t}
}()

// sglangChatCompletionsProxy forwards the full JSON body to SGLang's OpenAI-compatible endpoint when
// video_url is present—partial rewriting would duplicate SGLang's parser and drift over time.
func (s *Server) sglangChatCompletionsProxy() gin.HandlerFunc {
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
		if !openai.ChatCompletionRequestHasVideoURL(&oreq) {
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

		name, err := getExistingName(modelRef.Name)
		if err != nil {
			c.Next()
			return
		}

		m, err := GetModel(name.String())
		if err != nil {
			c.Next()
			return
		}

		backend := ""
		if m.Config.ModalityBackends != nil {
			backend = m.Config.ModalityBackends[model.ModalityVideoUnderstanding]
		}
		base := envconfig.SGLangURL()
		if backend != model.BackendSGLang || base == "" {
			c.Next()
			return
		}

		target := strings.TrimSuffix(base, "/") + "/v1/chat/completions"
		outReq, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, target, bytes.NewReader(body))
		if err != nil {
			slog.Error("sglang proxy: build request", "error", err)
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

		resp, err := sglangProxyClient.Do(outReq)
		if err != nil {
			slog.Warn("sglang proxy: request failed", "error", err)
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
			slog.Debug("sglang proxy: copy response", "error", err)
		}
		c.Abort()
	}
}
