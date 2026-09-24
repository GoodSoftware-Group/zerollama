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

// DecisionsHandler serves POST /v1/decisions and POST /v1/systemone (Jev/Laya typed decisions).
// WHY a dedicated handler: not chat, not /api/score, not /v1/rerank — needs a Decider
// runner (llama-server --decisions). Alias /v1/systemone keeps Jev clients on one path.
func (s *Server) DecisionsHandler(c *gin.Context) {
	var req api.DecisionsRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	if len(req.Questions) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "questions required"})
		return
	}
	if req.State == nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "state is required"})
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
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "decisions is not available for cloud models"})
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

	decider, ok := r.(llm.Decider)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "model runner does not support decisions"})
		return
	}

	llmReq, err := apiDecisionsToLLM(req)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	resp, err := decider.Decisions(c.Request.Context(), llmReq)
	if err != nil {
		var se api.StatusError
		if errors.As(err, &se) {
			c.AbortWithStatusJSON(se.StatusCode, gin.H{"error": se.ErrorMessage})
			return
		}
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	out := api.DecisionsResponse{
		Model:   name.String(),
		Answers: resp.Answers,
	}
	if resp.Model != "" {
		out.Model = resp.Model
	}
	out.Usage.InputTokens = resp.Usage.InputTokens
	out.Usage.OutputTokens = resp.Usage.OutputTokens
	c.JSON(http.StatusOK, out)
}

func apiDecisionsToLLM(req api.DecisionsRequest) (llm.DecisionsRequest, error) {
	out := llm.DecisionsRequest{
		Model:     req.Model,
		State:     req.State,
		Questions: make(map[string]llm.DecisionQuestion, len(req.Questions)),
	}
	for id, q := range req.Questions {
		t := strings.ToLower(strings.TrimSpace(q.Type))
		if t != "choice" && t != "score" && t != "noul" {
			return llm.DecisionsRequest{}, fmt.Errorf("question %q: type must be choice|score|noul", id)
		}
		if strings.TrimSpace(q.Instructions) == "" {
			return llm.DecisionsRequest{}, fmt.Errorf("question %q: instructions required", id)
		}
		out.Questions[id] = llm.DecisionQuestion{
			Type:         t,
			Instructions: q.Instructions,
			Criteria:     q.Criteria,
			Labels:       q.Labels,
		}
	}
	return out, nil
}
