package server

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/version"
)

// VersionHandler serves GET/HEAD /api/version with build metadata for operators and smokes.
// WHY edge_build separate from runtime edge: -tags edge binaries may auto-apply serve defaults;
// fleet needs to know compile artifact vs env-only --edge without shelling out to `zerollama -v`.
func VersionHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"version":      version.Version,
		"distribution": "zerollama",
		"edge_build":   version.IsEdgeBuild(),
		"zerollama": gin.H{
			"capabilities": zerollamaVersionCapabilities(),
			"qos":          zerollamaVersionQoS(),
		},
	})
}
