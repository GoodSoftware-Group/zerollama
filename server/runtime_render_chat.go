package server

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/tools"
)

type renderChatRequest struct {
	Model     string          `json:"model"`
	Messages  []api.Message   `json:"messages"`
	Tools     api.Tools       `json:"tools"`
	Think     *api.ThinkValue `json:"think"`
	NumCtx     int             `json:"num_ctx"`
	NumPredict int             `json:"num_predict"`
	Truncate   *bool           `json:"truncate"`
}

// RenderChatHandler renders a chat prompt using the model Modelfile template or builtin renderer.
func (s *Server) RenderChatHandler(c *gin.Context) {
	var req renderChatRequest
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
	truncate := req.Truncate == nil || *req.Truncate
	prepMsgs := prepareRenderMessages(m, req.Messages)
	prompt, truncateMode, droppedPrefix, hasToolSupport, err := s.renderChatPromptPrepared(
		c.Request.Context(),
		m,
		prepMsgs,
		req.Tools,
		req.Think,
		req.NumCtx,
		req.NumPredict,
		truncate,
	)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	tag := "{"
	if m.Template != nil && m.Template.Template != nil {
		tag = tools.TemplateToolTag(m.Template.Template)
	}
	c.JSON(http.StatusOK, gin.H{
		"prompt":           prompt,
		"tool_tag":         tag,
		"renderer":         strings.TrimSpace(m.Config.Renderer),
		"template":         m.Template != nil && m.Template.Template != nil,
		"parser":           strings.TrimSpace(m.Config.Parser),
		"has_tool_support": hasToolSupport,
		"truncated":        droppedPrefix,
		"truncate_mode":    truncateMode,
	})
}
