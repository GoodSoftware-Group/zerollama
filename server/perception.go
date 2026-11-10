package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
)

// perceptionClient bounds a cold OWLv2 or SAM2 forward. The first load is the
// slow call; later clips reuse the sibling.
var perceptionClient = &http.Client{Timeout: 3 * time.Minute}

const maxPerceptionBody = 20 << 20

// DetectHandler serves POST /v1/detect. With ZEROLLAMA_DETECT_URL unset the
// route still exists and answers status=wire_only, so a client that parsed
// OpenAPI does not receive gin's HTML 404.
func (s *Server) DetectHandler(c *gin.Context) {
	s.perceptionProxy(c, envconfig.DetectURL(), "/v1/detect", map[string]any{
		"status":     "wire_only",
		"model":      "owlv2",
		"detections": []any{},
		"reason":     "ZEROLLAMA_DETECT_URL is unset",
	})
}

// SegmentHandler serves POST /v1/segment. grasp_ready stays false on the
// stub: a mask, once the sibling exists, is still not a grasp.
func (s *Server) SegmentHandler(c *gin.Context) {
	s.perceptionProxy(c, envconfig.SegmentURL(), "/v1/segment", map[string]any{
		"status":      "wire_only",
		"grasp_ready": false,
		"mask":        nil,
		"reason":      "ZEROLLAMA_SEGMENT_URL is unset",
	})
}

func (s *Server) perceptionProxy(c *gin.Context, base, path string, stub map[string]any) {
	body, err := io.ReadAll(io.LimitReader(c.Request.Body, maxPerceptionBody+1))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "read body: " + err.Error()})
		return
	}
	if len(body) > maxPerceptionBody {
		c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{"error": "image body exceeds 20MB"})
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
	if !perceptionHasImage(raw) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "image, image_base64, image_url, or frame is required"})
		return
	}
	if base == "" {
		c.JSON(http.StatusOK, stub)
		return
	}
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, base+path, bytes.NewReader(body))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := perceptionClient.Do(req)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "perception upstream: " + err.Error()})
		return
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(io.LimitReader(resp.Body, maxPerceptionBody))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "perception read: " + err.Error()})
		return
	}
	ctype := resp.Header.Get("Content-Type")
	if ctype == "" {
		ctype = "application/json"
	}
	c.Data(resp.StatusCode, ctype, out)
}

func perceptionHasImage(raw map[string]any) bool {
	for _, key := range []string{"image", "image_base64", "image_url", "frame"} {
		s, _ := raw[key].(string)
		if strings.TrimSpace(s) != "" {
			return true
		}
	}
	return false
}
