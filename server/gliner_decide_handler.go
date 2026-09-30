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

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// glinerDecideClient bounds Fastino Decide sibling round-trips (cold load can be slow).
var glinerDecideClient = &http.Client{
	Timeout: 10 * time.Minute,
}

func glinerDecideBackend(cfg model.ConfigV2) string {
	return modality.BackendFor(cfg, model.ModalityDecisions)
}

// glinerDecideExternalResolve returns ZEROLLAMA_GLINER_DECIDE_URL when the request should proxy.
func glinerDecideExternalResolve(modelName string, m *Model) (string, error) {
	if m != nil && glinerDecideBackend(m.Config) == model.BackendGlinerDecide {
		if u := envconfig.GlinerDecideURL(); u != "" {
			return u, nil
		}
		return "", fmt.Errorf("modality_backends.decisions=gliner-decide requires ZEROLLAMA_GLINER_DECIDE_URL; see docs/gliner-decide.md")
	}
	if isGlinerDecideModelName(modelName) {
		if u := envconfig.GlinerDecideURL(); u != "" {
			return u, nil
		}
		return "", fmt.Errorf("model %q requires ZEROLLAMA_GLINER_DECIDE_URL; see docs/gliner-decide.md", modelName)
	}
	if u := envconfig.GlinerDecideURL(); u != "" && strings.TrimSpace(modelName) != "" {
		return u, nil
	}
	return "", fmt.Errorf("set ZEROLLAMA_GLINER_DECIDE_URL for /v1/gliner-decide; see docs/gliner-decide.md")
}

// GlinerDecideHandler serves POST /v1/gliner-decide (mechanical classify_text schema).
func (s *Server) GlinerDecideHandler(c *gin.Context) {
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
		modelName = "gliner-decide"
		raw["model"] = modelName
	}
	var m *Model
	if name, err := getExistingName(model.ParseName(modelName)); err == nil {
		if mm, err := GetModel(name.String()); err == nil {
			m = mm
		}
	}
	base, err := glinerDecideExternalResolve(modelName, m)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	fwd, err := json.Marshal(raw)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	target := strings.TrimSuffix(base, "/") + "/v1/gliner-decide"
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, target, bytes.NewReader(fwd))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := glinerDecideClient.Do(req)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "gliner-decide proxy: " + err.Error()})
		return
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "gliner-decide proxy: read: " + err.Error()})
		return
	}
	c.Data(resp.StatusCode, "application/json", out)
}
