package server

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/internal/runtimeclient"
)

// RuntimeKVSnapshotHandler proxies loopback GET /internal/kv-snapshot to the Python runtime.
func (s *Server) RuntimeKVSnapshotHandler(c *gin.Context) {
	body, status, err := runtimeclient.FetchKVSnapshot(c.Request.Context())
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	if status == http.StatusServiceUnavailable {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
			"error": "python runtime not configured",
		})
		return
	}
	c.Data(status, "application/json", body)
	c.Abort()
}
