package openapi

import (
	_ "embed"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/version"
)

//go:embed openapi.yaml
var rawYAML []byte

//go:embed docs.html
var docsHTML []byte

var (
	docMu   sync.Mutex
	baseDoc map[string]any
)

func loadBase() (map[string]any, error) {
	docMu.Lock()
	defer docMu.Unlock()
	if baseDoc != nil {
		return baseDoc, nil
	}
	var doc map[string]any
	if err := yaml.Unmarshal(rawYAML, &doc); err != nil {
		return nil, err
	}
	baseDoc = doc
	return baseDoc, nil
}

// Document returns a deep-ish copy of the OpenAPI doc with server URL and version filled in.
func Document(serverURL string) (map[string]any, error) {
	base, err := loadBase()
	if err != nil {
		return nil, err
	}
	doc := cloneMap(base)
	if info, ok := doc["info"].(map[string]any); ok {
		info["version"] = version.Version
		info["title"] = "zerollama API"
	}
	if serverURL != "" {
		doc["servers"] = []any{
			map[string]any{
				"url":         strings.TrimRight(serverURL, "/"),
				"description": "This zerollama server",
			},
		}
	}
	return doc, nil
}

func cloneMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		switch t := v.(type) {
		case map[string]any:
			out[k] = cloneMap(t)
		case []any:
			out[k] = cloneSlice(t)
		default:
			out[k] = v
		}
	}
	return out
}

func cloneSlice(in []any) []any {
	out := make([]any, len(in))
	for i, v := range in {
		switch t := v.(type) {
		case map[string]any:
			out[i] = cloneMap(t)
		case []any:
			out[i] = cloneSlice(t)
		default:
			out[i] = v
		}
	}
	return out
}

func requestBaseURL(c *gin.Context) string {
	scheme := "http"
	if c.Request.TLS != nil || strings.EqualFold(c.GetHeader("X-Forwarded-Proto"), "https") {
		scheme = "https"
	}
	host := c.Request.Host
	if host == "" {
		host = "localhost"
	}
	return scheme + "://" + host
}

// YAMLHandler serves GET /openapi.yaml with the live server URL injected.
func YAMLHandler(c *gin.Context) {
	doc, err := Document(requestBaseURL(c))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	out, err := yaml.Marshal(doc)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.Header("Cache-Control", "no-cache")
	c.Data(http.StatusOK, "application/yaml; charset=utf-8", out)
}

// JSONHandler serves GET /openapi.json with the live server URL injected.
func JSONHandler(c *gin.Context) {
	doc, err := Document(requestBaseURL(c))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.Header("Cache-Control", "no-cache")
	c.JSON(http.StatusOK, doc)
}

// DocsHandler serves GET /docs — Swagger UI pointed at this server's /openapi.json.
func DocsHandler(c *gin.Context) {
	html := strings.ReplaceAll(string(docsHTML), "{{VERSION}}", version.Version)
	c.Header("Cache-Control", "no-cache")
	c.Data(http.StatusOK, "text/html; charset=utf-8", []byte(html))
}

// Register mounts /openapi.yaml, /openapi.json, and /docs on r.
func Register(r gin.IRoutes) {
	r.GET("/openapi.yaml", YAMLHandler)
	r.GET("/openapi.json", JSONHandler)
	r.GET("/docs", DocsHandler)
	r.GET("/docs/", DocsHandler)
}

// SpecSummary is a one-line pointer used on the root handler.
func SpecSummary() string {
	return fmt.Sprintf("zerollama %s — docs /docs · openapi /openapi.json", version.Version)
}
