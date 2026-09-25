package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

// DecisionsHandler serves POST /v1/decisions and POST /v1/systemone (Jev/Laya/CLM typed decisions).
// WHY a dedicated handler: not chat, not /api/score, not /v1/rerank — needs a Decider
// runner (llama-server --decisions) or an external URL (ZEROLLAMA_CLM_URL / ZEROLLAMA_LAYA_URL).
// Alias /v1/systemone keeps Jev/CLM clients on one path.
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

	// Optional local model for modality_backends.decisions=clm|laya. CLM name
	// heuristics work without a local tag when ZEROLLAMA_CLM_URL is set.
	var m *Model
	name, nameErr := getExistingName(modelRef.Name)
	if nameErr == nil {
		if loaded, err := GetModel(name.String()); err == nil {
			m = loaded
		}
	}

	if base, err := decisionsExternalResolve(req.Model, m); err != nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	} else if base != "" {
		out, err := proxyDecisionsExternal(c.Request.Context(), base, req.Model, m, req)
		if err != nil {
			var se api.StatusError
			if errors.As(err, &se) {
				c.AbortWithStatusJSON(se.StatusCode, gin.H{"error": se.ErrorMessage})
				return
			}
			c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, out)
		return
	}

	if clmWantsNative(req.Model, m) {
		llmReq, err := apiDecisionsToLLM(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		resp, err := llm.CLMAnswer(c.Request.Context(), envconfig.CLMHeads(), envconfig.CLMEmbURL(), envconfig.CLMEmbModel(), llmReq, 1)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": err.Error()})
			return
		}
		out := api.DecisionsResponse{
			Model:   resp.Model,
			Answers: resp.Answers,
		}
		out.Usage.InputTokens = resp.Usage.InputTokens
		out.Usage.OutputTokens = resp.Usage.OutputTokens
		c.JSON(http.StatusOK, out)
		return
	}

	if nameErr != nil {
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
