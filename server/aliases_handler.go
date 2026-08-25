package server

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

func (s *Server) AliasesHandler(c *gin.Context) {
	if c.Request.Method == http.MethodGet {
		list, err := listAliases()
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if list == nil {
			list = []AliasInfo{}
		}
		c.JSON(http.StatusOK, list)
		return
	}

	var req struct {
		Name   string `json:"name"`
		Target string `json:"target"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	name := strings.TrimSpace(req.Name)
	target := strings.TrimSpace(req.Target)
	if name == "" || target == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "name and target are required"})
		return
	}
	if name == target {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "alias cannot point to itself"})
		return
	}
	m, err := loadAliasMap()
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if _, chained := m[target]; chained {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "alias target is itself an alias (chains are not allowed)"})
		return
	}
	addAliasOverlay(name, target)
	c.JSON(http.StatusOK, AliasInfo{Name: name, Target: target})
}
