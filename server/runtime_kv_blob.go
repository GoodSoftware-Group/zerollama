package server

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// KvBlobHandler proxies GET /api/kv/blob/:digest to the Python runtime (L3-R10).
// Why public (not loopback-only): fleet peers on the LAN pull cold-node KV blobs
// by content digest; optional ZEROLLAMA_LMCACHE_BLOB_HTTP_TOKEN is enforced by runtime.
func (s *Server) KvBlobHandler(c *gin.Context) {
	base := strings.TrimSpace(effectiveRuntimeURL())
	if base == "" {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
			"error": "python runtime not configured",
		})
		return
	}
	digest := strings.TrimSpace(strings.ToLower(c.Param("digest")))
	if len(digest) != 64 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid digest"})
		return
	}
	for _, ch := range digest {
		if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid digest"})
			return
		}
	}

	url := strings.TrimSuffix(base, "/") + "/kv/blob/" + digest
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, url, nil)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	// Forward optional blob auth from peer puller.
	if auth := c.GetHeader("Authorization"); auth != "" {
		req.Header.Set("Authorization", auth)
	}
	if tok := c.GetHeader("X-Zerollama-Blob-Token"); tok != "" {
		req.Header.Set("X-Zerollama-Blob-Token", tok)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{
			"error": fmt.Sprintf("runtime blob fetch: %v", err),
		})
		return
	}
	defer resp.Body.Close()

	if dig := resp.Header.Get("X-Zerollama-Blob-Digest"); dig != "" {
		c.Header("X-Zerollama-Blob-Digest", dig)
	}
	c.Status(resp.StatusCode)
	c.Header("Content-Type", "application/octet-stream")
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if len(body) > 0 {
			c.Writer.Write(body)
		}
		c.Abort()
		return
	}
	_, _ = io.Copy(c.Writer, resp.Body)
	c.Abort()
}
