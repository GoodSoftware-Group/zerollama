package server

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

type routerDecideRequest struct {
	Router string `json:"router"`
	Prompt string `json:"prompt"`
	Input  string `json:"input"`
}

func (s *Server) scoreRouterPolicies(ctx context.Context, classifier, prompt string, candidates []string, lengthNorm bool) ([]llm.CandidateScore, error) {
	r, _, _, _, releaseQoS, err := s.scheduleRunner(ctx, classifier, []model.Capability{}, nil, nil, nil, nil, nil)
	if err != nil {
		return nil, err
	}
	defer releaseQoS()
	scorer, ok := r.(llm.Scorer)
	if !ok {
		return nil, fmt.Errorf("classifier %q does not support score", classifier)
	}
	resp, err := scorer.Score(ctx, llm.ScoreRequest{
		Prompt:          prompt,
		Candidates:      candidates,
		LengthNormalize: lengthNorm,
	})
	if err != nil {
		return nil, err
	}
	return resp.Candidates, nil
}

func (s *Server) RouterDecideHandler(c *gin.Context) {
	var req routerDecideRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	name := strings.TrimSpace(req.Router)
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		prompt = strings.TrimSpace(req.Input)
	}
	if name == "" || prompt == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "router and prompt are required"})
		return
	}
	spec, ok := lookupRouter(name)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("router %q not found", name)})
		return
	}
	dec, err := decideRouter(c.Request.Context(), name, spec, prompt, s.scoreRouterPolicies, s.embedRouterText, s.rerankRouterDocs)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, dec)
}

func (s *Server) applyRouterRewrite(ctx context.Context, modelName, prompt string) (string, *RouterDecision, error) {
	if !envconfig.RouterRewrite() {
		return modelName, nil, nil
	}
	spec, ok := lookupRouter(modelName)
	if !ok {
		return modelName, nil, nil
	}
	dec, err := decideRouter(ctx, modelName, spec, prompt, s.scoreRouterPolicies, s.embedRouterText, s.rerankRouterDocs)
	if err != nil {
		return "", nil, err
	}
	if dec.Candidate == "" {
		return "", nil, fmt.Errorf("router %q produced empty candidate", modelName)
	}
	return dec.Candidate, &dec, nil
}

func stampRouterHeaders(c *gin.Context, dec *RouterDecision) {
	if dec == nil {
		return
	}
	c.Header("X-Zerollama-Router", dec.Router)
	c.Header("X-Zerollama-Router-Chosen", dec.Candidate)
	if dec.Fallback {
		c.Header("X-Zerollama-Router-Fallback", "true")
	}
}

func (s *Server) embedRouterText(ctx context.Context, embedder, text string) ([]float32, error) {
	r, _, _, _, releaseQoS, err := s.scheduleRunner(ctx, embedder, []model.Capability{}, nil, nil, nil, nil, nil)
	if err != nil {
		return nil, err
	}
	defer releaseQoS()
	emb, _, err := r.Embedding(ctx, text)
	return emb, err
}

func (s *Server) RouterCorpusHandler(c *gin.Context) {
	routerName := strings.TrimSpace(c.Query("router"))
	if c.Request.Method == http.MethodGet {
		if routerName == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "router query is required"})
			return
		}
		spec, ok := lookupRouter(routerName)
		if !ok {
			c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("router %q not found", routerName)})
			return
		}
		total, byLabel := knnCorpusStats(routerName, spec)
		c.JSON(http.StatusOK, gin.H{"router": routerName, "total": total, "labels": byLabel})
		return
	}

	var req struct {
		Router  string              `json:"router"`
		Entries []RouterCorpusEntry `json:"entries"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	routerName = strings.TrimSpace(req.Router)
	spec, ok := lookupRouter(routerName)
	if !ok {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("router %q not found", routerName)})
		return
	}
	if !routerIsKNN(spec) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "router is not classifier knn"})
		return
	}
	added := addKNNCorpus(routerName, req.Entries)
	total, byLabel := knnCorpusStats(routerName, spec)
	c.JSON(http.StatusOK, gin.H{"router": routerName, "added": added, "total": total, "labels": byLabel})
}

func lastUserMessageText(msgs []api.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if strings.EqualFold(msgs[i].Role, "user") && strings.TrimSpace(msgs[i].Content) != "" {
			return msgs[i].Content
		}
	}
	return ""
}
