package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/server/autoroute"
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

// RouterDecideHandler serves POST /api/router/decide — the LA11 dry run ({router, prompt}).
// The prompt is treated as a plain chat turn; /v1/route accepts full request bodies.
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
	sig := autoroute.Signals{Surface: autoroute.SurfaceChat, Text: prompt, PromptTokens: estimateTokens(len(prompt))}
	dec, isRouter, err := s.resolveRouter(c.Request.Context(), name, sig)
	if !isRouter {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("router %q not found", name)})
		return
	}
	if err != nil {
		writeRouterError(c, dec, err)
		return
	}
	c.JSON(http.StatusOK, dec)
}

// resolveRouter answers "which model would this router pick for sig" without side effects.
// isRouter is false when name is neither a router.yaml router nor the implicit "auto", in
// which case the caller treats name as an ordinary model. On error dec may still be
// non-nil and carry the filtered candidates (implicit "auto"), so callers can explain.
//
// WHY one function: the in-band rewrite, /v1/route and /api/router/decide must agree —
// a dry run that differs from what chat actually does is worse than none.
func (s *Server) resolveRouter(ctx context.Context, name string, sig autoroute.Signals) (dec *RouterDecision, isRouter bool, err error) {
	return s.resolveRouterMem(ctx, name, sig, false)
}

// resolveRouterUncached is the routing decision itself, without sticky / cache memory.
func (s *Server) resolveRouterUncached(ctx context.Context, name string, sig autoroute.Signals) (dec *RouterDecision, isRouter bool, err error) {
	if spec, ok := lookupRouter(name); ok {
		if !sig.Surface.IsText() && sig.Surface != autoroute.SurfaceImage && sig.Surface != autoroute.SurfaceImageEdit {
			d, err := s.decideRouterNonText(ctx, name, spec, sig)
			return d, true, err
		}
		if routerIsJev(spec) && len(spec.Questions) > 0 && sig.Surface.IsText() {
			d, err := s.decideRouterJevMulti(ctx, name, spec, sig)
			if errors.Is(err, errDeciderUnavailable) && !deciderFailsClosed(spec) {
				fd, ferr := s.decideRouterNonText(ctx, name, spec, sig)
				if fd != nil && ferr == nil {
					fd.Reason, fd.Classifier = "decider-fallback", "jev"
					if d != nil {
						fd.Decider = d.Decider
					}
				}
				return fd, true, ferr
			}
			return d, true, err
		}
		d, err := decideRouterWith(ctx, name, spec, sig.Text, routerBackends{
			score: s.scoreRouterPolicies, embed: s.embedRouterText, rerank: s.rerankRouterDocs, jev: s.askJevDecider,
		})
		if errors.Is(err, errDeciderUnavailable) && !deciderFailsClosed(spec) {
			// Fail open (AR4): no router fallback configured, so pick deterministically
			// among the router's own candidates rather than 503 because the decider is cold.
			fd, ferr := s.decideRouterNonText(ctx, name, spec, sig)
			if fd != nil && ferr == nil {
				fd.Reason, fd.Decider = "decider-fallback", d.Decider
			}
			return fd, true, ferr
		}
		if err != nil {
			return nil, true, err
		}
		if d.Candidate == "" {
			return nil, true, fmt.Errorf("router %q produced empty candidate", name)
		}
		if err := enforceRouterSurface(&d, spec, sig); err != nil {
			return &d, true, err
		}
		return &d, true, nil
	}
	if isImplicitAutoName(name) {
		d, err := s.decideAuto(ctx, name, sig, nil)
		return d, true, err
	}
	return nil, false, nil
}

// decideRouterNonText resolves a router.yaml router on a non-text surface (speech, video,
// embed, decisions, …). WHY not the classifier: LA11 policies label chat intents; scoring a
// TTS input or an embedding batch against them is meaningless. Instead the router's
// embedder / reranker pins embed / rerank, and other surfaces pick deterministically among
// the router's own candidates and fallback.
func (s *Server) decideRouterNonText(ctx context.Context, name string, spec RouterSpec, sig autoroute.Signals) (*RouterDecision, error) {
	pin := ""
	switch sig.Surface {
	case autoroute.SurfaceEmbed:
		pin = spec.Embedder
	case autoroute.SurfaceRerank:
		pin = spec.Reranker
	}
	allow := map[string]bool{}
	add := func(n string) {
		if n = strings.TrimSpace(n); n != "" {
			allow[model.ParseName(n).DisplayShortest()] = true
		}
	}
	if pin != "" {
		add(pin)
	} else {
		for _, c := range spec.Candidates {
			add(c.Model)
		}
		add(spec.Fallback)
	}
	dec, err := s.decideAuto(ctx, name, sig, allow)
	if dec != nil && err == nil && pin != "" {
		dec.Reason = "pinned"
	}
	return dec, err
}

// applyRouterRewrite resolves a router name to a concrete model in-band. router.yaml
// routers (LA11) run their classifier and are then checked against the request's surface
// and capabilities; the reserved name "auto" without a router.yaml entry uses the built-in
// deterministic router (docs/auto-router.md AR1). Other names pass through unchanged.
func (s *Server) applyRouterRewrite(ctx context.Context, modelName string, sig autoroute.Signals) (string, *RouterDecision, error) {
	if !envconfig.RouterRewrite() {
		return modelName, nil, nil
	}
	dec, isRouter, err := s.resolveRouterMem(ctx, modelName, sig, true)
	if !isRouter {
		return modelName, nil, nil
	}
	if err != nil {
		metricsRouteError(modelName, sig.Surface)
		return "", nil, err
	}
	metricsRouteDecision(dec, sig.Surface)
	return dec.Candidate, dec, nil
}

// writeRouterError writes a 503 for a failed dry run, including the filtered candidates
// when the deterministic router produced them.
func writeRouterError(c *gin.Context, dec *RouterDecision, err error) {
	body := gin.H{"error": err.Error()}
	if dec != nil && dec.Auto != nil && len(dec.Auto.Rejected) > 0 {
		body["filtered"] = dec.Auto.Rejected
	}
	c.AbortWithStatusJSON(http.StatusServiceUnavailable, body)
}

func stampRouterHeaders(c *gin.Context, dec *RouterDecision) {
	if dec == nil {
		return
	}
	c.Header("X-Zerollama-Router", dec.Router)
	c.Header("X-Zerollama-Router-Chosen", dec.Candidate)
	if dec.StickyBreak != "" {
		c.Header("X-Zerollama-Router-Sticky-Break", dec.StickyBreak)
	}
	if sug := dec.Suggestion; sug != nil {
		c.Header("X-Zerollama-Route-Suggest", sug.Modality)
		c.Header("X-Zerollama-Route-Suggest-Endpoint", sug.Endpoint)
		if sug.Model != "" {
			c.Header("X-Zerollama-Route-Suggest-Model", sug.Model)
		}
	}
	if dec.Fallback {
		c.Header("X-Zerollama-Router-Fallback", "true")
	}
	if dec.Reason != "" {
		c.Header("X-Zerollama-Router-Reason", dec.Reason)
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
