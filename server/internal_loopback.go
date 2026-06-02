package server

import (
	"net"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
)

// ensureLoopbackGoURLEnv sets ZEROLLAMA_GO_URL for embedded runtime → Go /internal/* clients
// when OLLAMA_HOST binds 0.0.0.0 (ConnectableHost maps to 127.0.0.1).
func ensureLoopbackGoURLEnv() {
	if strings.TrimSpace(os.Getenv("ZEROLLAMA_GO_URL")) != "" {
		return
	}
	u := envconfig.ConnectableHost()
	_ = os.Setenv("ZEROLLAMA_GO_URL", strings.TrimSuffix(u.String(), "/"))
}

// internalLoopbackOnly rejects non-loopback callers (defense in depth for /internal/* helpers).
func internalLoopbackOnly() gin.HandlerFunc {
	return func(c *gin.Context) {
		host := c.ClientIP()
		if h, _, err := net.SplitHostPort(host); err == nil {
			host = h
		}
		if !isLoopbackHost(strings.TrimSpace(host)) {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error": "internal endpoints are loopback-only",
			})
			return
		}
		c.Next()
	}
}
