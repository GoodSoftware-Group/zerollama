package server

import (
	"bytes"
	"context"
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
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

// DecisionsHandler serves POST /v1/decisions and POST /v1/systemone (Jev/Laya/CLM/OpenJev).
//
// WHY a dedicated handler: not chat, not /api/score, not /v1/rerank — needs a Decider
// runner (llama-server --decisions) or an external URL (ZEROLLAMA_CLM_URL / ZEROLLAMA_LAYA_URL /
// ZEROLLAMA_OPENJEV_URL for DiffusionGemma / ZEROLLAMA_GLINER_DECIDE_URL for GLiNER2.5-Decide).
// Alias /v1/systemone keeps Jev clients on one path.
//
// WHY the handler is thin: backend dispatch lives in runDecisions so the auto-router can ask
// a local decider the same typed questions in-process (docs/auto-router.md AR0/AR4).
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

	call, err := parseDecisionsCall(body)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if served, err := applyModelAlias(c, call.Req.Model); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else {
		call.Req.Model = served
		call.LLM.Model = served
	}

	out, err := s.runDecisions(c.Request.Context(), call)
	if err != nil {
		writeDecisionsError(c, err)
		return
	}
	c.JSON(http.StatusOK, out)
}

// decisionsCall is a validated decisions request ready for backend dispatch.
type decisionsCall struct {
	Req api.DecisionsRequest
	LLM llm.DecisionsRequest
	// Body is the raw JSON. WHY keep it: api.DecisionsRequest.Questions is a map, so
	// re-marshalling loses question insertion order, which score encodings depend on.
	Body []byte
}

// parseDecisionsCall validates a decisions body (all failures are 400s) and enforces
// the Jev/Unsloth caps (64 questions / 255 choice options / 10 score levels) before
// any backend — Laya, CLM, OpenJev, or external URL.
func parseDecisionsCall(body []byte) (decisionsCall, error) {
	var req api.DecisionsRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return decisionsCall{}, err
	}
	if strings.TrimSpace(req.Model) == "" {
		return decisionsCall{}, errors.New("model is required")
	}
	if len(req.Questions) == 0 {
		return decisionsCall{}, errors.New("questions required")
	}
	if req.State == nil {
		return decisionsCall{}, errors.New("state is required")
	}
	llmReq, err := apiDecisionsToLLM(req)
	if err != nil {
		return decisionsCall{}, err
	}
	return decisionsCall{Req: req, LLM: llmReq, Body: body}, nil
}

// decisionsError carries the HTTP status runDecisions wants written. Schedule failures
// set scheduleModel instead so the handler keeps handleScheduleError's mapping
// (Retry-After, 499 on cancel, preempted_reason) without runDecisions touching gin.
type decisionsError struct {
	status        int
	err           error
	scheduleModel string
}

func (e *decisionsError) Error() string { return e.err.Error() }
func (e *decisionsError) Unwrap() error { return e.err }

func decisionsErr(status int, err error) error {
	return &decisionsError{status: status, err: err}
}

func decisionsScheduleErr(model string, err error) error {
	return &decisionsError{err: err, scheduleModel: model}
}

// decisionsStatusErr preserves a backend api.StatusError (code + bare message) or falls
// back to the given status.
func decisionsStatusErr(fallback int, err error) error {
	var se api.StatusError
	if errors.As(err, &se) {
		return decisionsErr(se.StatusCode, errors.New(se.ErrorMessage))
	}
	return decisionsErr(fallback, err)
}

// decisionsModelRefErr mirrors writeModelRefParseError with a 404 fallback.
func decisionsModelRefErr(err error, modelName string) error {
	switch {
	case errors.Is(err, errConflictingModelSource):
		return decisionsErr(http.StatusBadRequest, err)
	case errors.Is(err, model.ErrUnqualifiedName):
		return decisionsErr(http.StatusBadRequest, errors.New(errtypes.InvalidModelNameErrMsg))
	default:
		return decisionsErr(http.StatusNotFound, fmt.Errorf("model '%s' not found", modelName))
	}
}

func writeDecisionsError(c *gin.Context, err error) {
	var de *decisionsError
	if !errors.As(err, &de) {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if de.scheduleModel != "" {
		handleScheduleError(c, de.scheduleModel, de.err)
		return
	}
	c.AbortWithStatusJSON(de.status, gin.H{"error": de.err.Error()})
}

// runDecisions dispatches a validated call (model already alias-resolved) to the right
// decider: external URL proxy, native CLM heads, System One score models, or a Decider
// runner (Laya GGUF). Errors are *decisionsError.
//
// WHY OpenJev branches before scheduleRunner: sibling diffusion server is not a ggml Model
// runner; Go packs prompt and proxies (LA16). Answers are calibrated:false until marker logits.
func (s *Server) runDecisions(ctx context.Context, call decisionsCall) (api.DecisionsResponse, error) {
	req, llmReq := call.Req, call.LLM

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		return api.DecisionsResponse{}, decisionsModelRefErr(err, req.Model)
	}
	if modelRef.Source == modelSourceCloud {
		return api.DecisionsResponse{}, decisionsErr(http.StatusNotImplemented, errors.New("decisions is not available for cloud models"))
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
		return api.DecisionsResponse{}, decisionsErr(http.StatusServiceUnavailable, err)
	} else if base != "" {
		out, err := proxyDecisionsExternal(ctx, base, req.Model, m, req)
		if err != nil {
			return api.DecisionsResponse{}, decisionsStatusErr(http.StatusBadGateway, err)
		}
		return out, nil
	}

	if clmWantsNative(req.Model, m) {
		resp, err := llm.CLMAnswer(ctx, envconfig.CLMHeads(), envconfig.CLMEmbURL(), envconfig.CLMEmbModel(), llmReq, 1)
		if err != nil {
			return api.DecisionsResponse{}, decisionsErr(http.StatusBadGateway, err)
		}
		out := api.DecisionsResponse{
			Model:   resp.Model,
			Answers: resp.Answers,
		}
		out.Usage.InputTokens = resp.Usage.InputTokens
		out.Usage.OutputTokens = resp.Usage.OutputTokens
		return out, nil
	}

	if nameErr != nil {
		return api.DecisionsResponse{}, decisionsErr(http.StatusNotFound, fmt.Errorf("model '%s' not found", req.Model))
	}

	if m != nil && modelUsesSystemOneScore(m) {
		dreq, err := unmarshalDecisionRequest(call.Body, req)
		if err != nil {
			return api.DecisionsResponse{}, decisionsErr(http.StatusBadRequest, err)
		}
		return s.decisionsSystemOneScore(ctx, m, name.String(), dreq, req.Options, req.KeepAlive)
	}

	if len(req.Images) > 0 {
		// Laya Decider cannot consume images; score-path models should have been
		// selected above. Fail clearly instead of ignoring multimodal state.
		return api.DecisionsResponse{}, decisionsErr(http.StatusBadRequest, errors.New(
			"images require a multimodal decision model (clef/tev1/nimble); Laya and text-only Deciders reject images"))
	}

	caps := []model.Capability{}
	if m != nil && capabilityContains(m.Config.Capabilities, model.CapabilityDecision) {
		caps = append(caps, model.CapabilityDecision)
	}

	r, _, _, _, releaseQoS, err := s.scheduleRunner(ctx, name.String(), caps, req.Options, req.KeepAlive, nil, nil, nil)
	if err != nil {
		return api.DecisionsResponse{}, decisionsScheduleErr(req.Model, err)
	}
	defer releaseQoS()

	decider, ok := r.(llm.Decider)
	if !ok {
		return api.DecisionsResponse{}, decisionsErr(http.StatusNotImplemented, errors.New("model runner does not support decisions"))
	}

	resp, err := decider.Decisions(ctx, llmReq)
	if err != nil {
		return api.DecisionsResponse{}, decisionsStatusErr(http.StatusInternalServerError, err)
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
	return out, nil
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
