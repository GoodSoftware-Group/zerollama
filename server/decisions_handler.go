package server

import (
	"bytes"
	"encoding/json"
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

// DecisionsHandler serves POST /v1/decisions and POST /v1/systemone (Jev/Laya/CLM/OpenJev).
//
// WHY a dedicated handler: not chat, not /api/score, not /v1/rerank — needs a Decider
// runner (llama-server --decisions) or an external URL (ZEROLLAMA_CLM_URL / ZEROLLAMA_LAYA_URL /
// ZEROLLAMA_OPENJEV_URL for DiffusionGemma / ZEROLLAMA_GLINER_DECIDE_URL for GLiNER2.5-Decide).
// Alias /v1/systemone keeps Jev clients on one path.
//
// WHY OpenJev branches before scheduleRunner: sibling diffusion server is not a ggml Model
// runner; Go packs prompt and proxies (LA16). Answers are calibrated:false until marker logits.
func (s *Server) DecisionsHandler(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 32<<20)
	body, err := io.ReadAll(c.Request.Body)
	if errors.Is(err, io.EOF) || len(bytes.TrimSpace(body)) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	}
	if err != nil {
		var sizeErr *http.MaxBytesError
		if errors.As(err, &sizeErr) {
			c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{"error": "request body must not exceed 32 MiB"})
			return
		}
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var req api.DecisionsRequest
	if err := json.Unmarshal(body, &req); err != nil {
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
	// Jev/Unsloth caps (64 questions / 255 choice options / 10 score levels) before
	// any backend — Laya, CLM, OpenJev, or external URL.
	llmReq, err := apiDecisionsToLLM(req)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if served, err := applyModelAlias(c, req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		req.Model = served
		llmReq.Model = served
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

	if m != nil && modelUsesSystemOneScore(m) {
		dreq, err := unmarshalDecisionRequest(body, req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		s.serveDecisionsSystemOneScore(c, m, name.String(), dreq, req.Options, req.KeepAlive)
		return
	}

	if len(req.Images) > 0 {
		// Laya Decider cannot consume images; score-path models should have been
		// selected above. Fail clearly instead of ignoring multimodal state.
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"error": "images require a multimodal decision model (clef/tev1/nimble); Laya and text-only Deciders reject images",
		})
		return
	}

	caps := []model.Capability{}
	if m != nil && capabilityContains(m.Config.Capabilities, model.CapabilityDecision) {
		caps = append(caps, model.CapabilityDecision)
	}

	r, _, _, _, releaseQoS, err := s.scheduleRunner(c.Request.Context(), name.String(), caps, req.Options, req.KeepAlive, nil, nil, nil)
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
		Images:    append([]api.ImageData(nil), req.Images...),
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
	if err := llm.ValidateDecisionBatch(out.Questions); err != nil {
		return llm.DecisionsRequest{}, err
	}
	return out, nil
}
