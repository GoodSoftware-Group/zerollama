package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func (s *Server) RerankHandler(c *gin.Context) {
	var req api.RerankRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	docs := req.Documents
	if len(docs) == 0 {
		docs = req.Texts
	}
	if len(docs) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "documents required"})
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "query is required"})
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	if served, err := applyModelAlias(c, req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		req.Model = served
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "rerank is not available for cloud models"})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer releaseQoS()

	reranker, ok := r.(llm.Reranker)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "model runner does not support rerank"})
		return
	}

	resp, err := reranker.Rerank(c.Request.Context(), llm.RerankRequest{
		Query:     req.Query,
		Documents: docs,
		TopN:      req.TopN,
	})
	if err != nil {
		var se api.StatusError
		if errors.As(err, &se) {
			c.AbortWithStatusJSON(se.StatusCode, gin.H{"error": se.ErrorMessage})
			return
		}
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	out := api.RerankResponse{
		Model:   name.String(),
		Object:  "list",
		Results: make([]api.RerankResult, 0, len(resp.Results)),
	}
	if resp.Model != "" {
		out.Model = resp.Model
	}
	out.Usage.PromptTokens = resp.PromptTokens
	out.Usage.TotalTokens = resp.TotalTokens
	for _, hit := range resp.Results {
		out.Results = append(out.Results, api.RerankResult{Index: hit.Index, RelevanceScore: hit.RelevanceScore})
	}
	c.JSON(http.StatusOK, out)
}
