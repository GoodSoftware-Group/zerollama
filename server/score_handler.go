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

func (s *Server) ScoreHandler(c *gin.Context) {
	var req api.ScoreRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if len(req.Candidates) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "candidates required"})
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "score is not available for cloud models"})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, _, _, _, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	scorer, ok := r.(llm.Scorer)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotImplemented, gin.H{"error": "model runner does not support score"})
		return
	}

	resp, err := scorer.Score(c.Request.Context(), llm.ScoreRequest{
		Prompt:               req.Prompt,
		Candidates:           req.Candidates,
		LengthNormalize:      req.LengthNormalize,
		IncludeTokenLogprobs: req.IncludeTokenLogprobs,
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

	c.JSON(http.StatusOK, api.ScoreResponse{
		Model:      name.String(),
		Candidates: scoreCandidatesToAPI(resp.Candidates),
	})
}

func scoreCandidatesToAPI(in []llm.CandidateScore) []api.CandidateScore {
	out := make([]api.CandidateScore, len(in))
	for i, c := range in {
		out[i] = api.CandidateScore{
			Candidate:               c.Candidate,
			LogProb:                 c.LogProb,
			LengthNormalizedLogProb: c.LengthNormalizedLogProb,
			NumTokens:               c.NumTokens,
		}
		if len(c.Tokens) > 0 {
			out[i].Tokens = make([]api.TokenLogprob, len(c.Tokens))
			for j, t := range c.Tokens {
				out[i].Tokens[j] = api.TokenLogprob{Token: t.Token, Logprob: t.Logprob}
			}
		}
	}
	return out
}
