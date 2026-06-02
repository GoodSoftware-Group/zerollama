package server

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

type parseToolOutputRequest struct {
	Model    string          `json:"model"`
	Messages []api.Message   `json:"messages"`
	Content  string          `json:"content"`
	Done     bool            `json:"done"`
	Tools    api.Tools       `json:"tools"`
	Think    *api.ThinkValue `json:"think"`
}

// ParseToolOutputHandler parses model output using the same logic as ChatHandler.
func (s *Server) ParseToolOutputHandler(c *gin.Context) {
	_ = s
	var req parseToolOutputRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	modelName := strings.TrimSpace(req.Model)
	if modelName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model required"})
		return
	}
	m, err := GetModel(modelName)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	id, method, err := toolParseSessions.open(m, req.Tools, req.Messages, req.Think)
	if err != nil {
		if err == errNoToolParser {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if err == errTooManyToolParseSessions {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer toolParseSessions.close(id)
	content, thinking, toolCalls, method, err := toolParseSessions.add(
		id,
		req.Content,
		req.Done,
	)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if method == "" {
		method = strings.TrimSpace(m.Config.Parser)
	}
	c.JSON(http.StatusOK, toolParseResponse(content, thinking, toolCalls, method))
}
