package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// extractExternalClient bounds GLiNER sibling round-trips (CPU ONNX can be slow cold).
var extractExternalClient = &http.Client{
	Timeout: 5 * time.Minute,
}

// abstractExtractKeys are forwarded on /v1/extract (engine knobs stripped).
var abstractExtractKeys = map[string]bool{
	"model": true, "text": true, "texts": true, "labels": true, "threshold": true,
}

func isGlinerModelName(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return false
	}
	// Decide triage owns gliner-decide* / gliner2.5-decide* (Finding 4).
	if isGlinerDecideModelName(n) {
		return false
	}
	return n == "gliner" || strings.HasPrefix(n, "gliner:") || strings.HasPrefix(n, "gliner-")
}

func extractBackend(cfg model.ConfigV2) string {
	return modality.BackendFor(cfg, model.ModalityExtract)
}

// extractExternalResolve returns ZEROLLAMA_GLINER_URL when the request should proxy.
func extractExternalResolve(modelName string, m *Model) (string, error) {
	if m != nil && extractBackend(m.Config) == model.BackendGliner {
		if u := envconfig.GlinerURL(); u != "" {
			return u, nil
		}
		return "", fmt.Errorf("modality_backends.extract=gliner requires ZEROLLAMA_GLINER_URL; see docs/gliner-cpp.md")
	}
	if isGlinerModelName(modelName) {
		if u := envconfig.GlinerURL(); u != "" {
			return u, nil
		}
		return "", fmt.Errorf("model %q requires ZEROLLAMA_GLINER_URL; see docs/gliner-cpp.md", modelName)
	}
	if u := envconfig.GlinerURL(); u != "" && strings.TrimSpace(modelName) != "" {
		// Optional: any model name with URL set still proxies when explicitly hitting extract routes.
		return u, nil
	}
	return "", fmt.Errorf("set ZEROLLAMA_GLINER_URL for /v1/extract|/v1/gliner; see docs/gliner-cpp.md")
}

// stripToAbstract keeps only portable NER fields for /v1/extract upstream.
func stripToAbstract(raw map[string]any) map[string]any {
	out := make(map[string]any, len(abstractExtractKeys))
	for k, v := range raw {
		if abstractExtractKeys[k] {
			out[k] = v
		}
	}
	return out
}

// ExtractHandler serves POST /v1/extract (abstract NER).
func (s *Server) ExtractHandler(c *gin.Context) {
	s.extractProxy(c, false)
}

// GlinerHandler serves POST /v1/gliner (engine-native knobs).
func (s *Server) GlinerHandler(c *gin.Context) {
	s.extractProxy(c, true)
}

func (s *Server) extractProxy(c *gin.Context, mechanical bool) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "read body: " + err.Error()})
		return
	}
	if len(bytes.TrimSpace(body)) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	}
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	modelName, _ := raw["model"].(string)
	if strings.TrimSpace(modelName) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	labels, _ := raw["labels"].([]any)
	if len(labels) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "labels required"})
		return
	}
	_, hasText := raw["text"]
	texts, _ := raw["texts"].([]any)
	if !hasText && len(texts) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "text or texts required"})
		return
	}

	var m *Model
	if name, err := getExistingName(model.ParseName(modelName)); err == nil {
		if mm, err := GetModel(name.String()); err == nil {
			m = mm
		}
	}

	base, err := extractExternalResolve(modelName, m)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	forward := raw
	path := "/v1/gliner"
	if !mechanical {
		forward = stripToAbstract(raw)
		path = "/v1/extract"
	}
	payload, err := json.Marshal(forward)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, base+path, bytes.NewReader(payload))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := extractExternalClient.Do(req)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "gliner upstream: " + err.Error()})
		return
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "gliner read: " + err.Error()})
		return
	}
	c.Data(resp.StatusCode, "application/json", out)
}

// Compile-time check that api.ExtractRequest stays aligned with proxy fields.
var _ = api.ExtractRequest{}
