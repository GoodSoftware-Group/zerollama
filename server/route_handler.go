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
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/types/model"
)

// routeEnvelope is the routing part of a POST /v1/route body. The rest of the body is the
// target surface's normal request (chat messages or a generate prompt), parsed separately.
type routeEnvelope struct {
	Router  string `json:"router"`
	Model   string `json:"model"`
	Surface string `json:"surface"`
}

// RouteHandler serves POST /v1/route — "which model would this request run on?" without
// running inference (docs/auto-router.md §10).
//
// WHY accept the real request body: agents can paste the exact chat/generate payload they
// are about to send (native or OpenAI-shaped) and get the decision the in-band rewrite would
// make, including why each other model was filtered.
func (s *Server) RouteHandler(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 32<<20)
	body, err := io.ReadAll(c.Request.Body)
	if err != nil || len(bytes.TrimSpace(body)) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	}
	var env routeEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	name := strings.TrimSpace(env.Router)
	if name == "" {
		name = strings.TrimSpace(env.Model)
	}
	if name == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "router (or model) is required"})
		return
	}
	if served, aliased, err := resolveAlias(name); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	} else if aliased {
		name = served
	}

	sig, err := routeSignalsFromBody(c, body, autoroute.Surface(strings.TrimSpace(env.Surface)))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	dec, isRouter, err := s.resolveRouter(c.Request.Context(), name, sig)
	if !isRouter {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("%q is not a router (router.yaml or implicit \"auto\")", name)})
		return
	}
	if err != nil {
		writeRouterError(c, dec, err)
		return
	}
	c.JSON(http.StatusOK, routeResponse{RouterDecision: *dec, Surface: sig.Surface, Model: dec.Candidate})
}

// routeResponse adds the resolved surface to RouterDecision; Model duplicates Candidate
// under the field name clients put in their next request.
type routeResponse struct {
	RouterDecision
	Surface autoroute.Surface `json:"surface"`
	Model   string            `json:"model"`
}

var routeSurfaces = map[autoroute.Surface]bool{
	autoroute.SurfaceChat: true, autoroute.SurfaceGenerate: true,
	autoroute.SurfaceImage: true, autoroute.SurfaceImageEdit: true,
	autoroute.SurfaceVideoGen: true, autoroute.SurfaceSpeech: true, autoroute.SurfaceMusic: true,
	autoroute.SurfaceTranscribe: true, autoroute.SurfaceEmbed: true, autoroute.SurfaceRerank: true,
	autoroute.SurfaceDecisions: true,
}

const routeSurfaceList = "chat, generate, image, image_edit, video_gen, speech, music, transcribe, embed, rerank, decisions"

// routeSignalsFromBody builds Signals from the target surface's normal request body, using
// the same signal builders as the handlers so a dry run cannot disagree with real routing.
// Surface is inferred when empty (messages ⇒ chat; width/height ⇒ image; otherwise
// generate); the other surfaces must be named because their bodies are ambiguous.
func routeSignalsFromBody(c *gin.Context, body []byte, surface autoroute.Surface) (autoroute.Signals, error) {
	if surface != "" && !routeSurfaces[surface] {
		return autoroute.Signals{}, fmt.Errorf("unsupported surface %q (%s)", surface, routeSurfaceList)
	}
	switch surface {
	case autoroute.SurfaceVideoGen:
		var req openai.VideoCreateRequest
		if err := json.Unmarshal(body, &req); err != nil {
			return autoroute.Signals{}, err
		}
		return videoRouteSignals(req), nil
	case autoroute.SurfaceSpeech, autoroute.SurfaceMusic:
		var req openai.SpeechCreateRequest
		if err := json.Unmarshal(body, &req); err != nil {
			return autoroute.Signals{}, err
		}
		return speechRouteSignals(req, surface), nil
	case autoroute.SurfaceTranscribe:
		// Same signals as TranscriptionHandler: the audio itself is never inspected.
		return autoroute.Signals{Surface: autoroute.SurfaceTranscribe, HasAudio: true}, nil
	case autoroute.SurfaceEmbed:
		return autoroute.Signals{Surface: autoroute.SurfaceEmbed}, nil
	case autoroute.SurfaceRerank:
		var req struct {
			Query string `json:"query"`
		}
		_ = json.Unmarshal(body, &req)
		return autoroute.Signals{Surface: autoroute.SurfaceRerank, Text: req.Query}, nil
	case autoroute.SurfaceDecisions:
		var req api.DecisionsRequest
		if err := json.Unmarshal(body, &req); err != nil {
			return autoroute.Signals{}, err
		}
		return decisionsRouteSignals(req), nil
	}
	var probe struct {
		Messages json.RawMessage `json:"messages"`
	}
	_ = json.Unmarshal(body, &probe)
	if surface == "" && len(probe.Messages) > 0 && !bytes.Equal(bytes.TrimSpace(probe.Messages), []byte("null")) {
		surface = autoroute.SurfaceChat
	}

	if surface == autoroute.SurfaceChat {
		req, err := parseRouteChat(body)
		if err != nil {
			return autoroute.Signals{}, err
		}
		sig := chatRouteSignals(c, req)
		// /api/chat and /v1/chat/completions route ahead of the runtime proxy. Dry runs
		// use this same builder, so they must admit sidecar models the live path can serve.
		sig.BeforeRuntimeProxy = true
		return sig, nil
	}

	var req api.GenerateRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return autoroute.Signals{}, err
	}
	if surface == autoroute.SurfaceImage || surface == autoroute.SurfaceImageEdit {
		c.Set(middleware.CtxKeyRouteSurface, string(surface))
	}
	sig := generateRouteSignals(c, &req)
	if sig.Surface.IsText() {
		sig.BeforeRuntimeProxy = true
	}
	return sig, nil
}

// parseRouteChat accepts native /api/chat bodies and OpenAI chat-completions bodies
// (content-part arrays, image_url, input_audio) so signals match what the handler sees.
//
// WHY prefer the OpenAI binder when OpenAI-only fields are present: a string-content
// /v1/chat/completions body also unmarshals into api.ChatRequest, which would drop
// functions, tool_choice, max_tokens and prompt_cache_key and mis-route (non-tool model
// for legacy functions, or tools required despite tool_choice:"none").
func parseRouteChat(body []byte) (*api.ChatRequest, error) {
	var oreq openai.ChatCompletionRequest
	oerr := json.Unmarshal(body, &oreq)
	if oerr == nil && openAIChatNeedsConversion(body, oreq) {
		return fromOpenAIForRoute(oreq)
	}
	var native api.ChatRequest
	nativeErr := json.Unmarshal(body, &native)
	if nativeErr == nil {
		return &native, nil
	}
	if oerr == nil {
		return fromOpenAIForRoute(oreq)
	}
	return nil, errors.Join(nativeErr, oerr)
}

// fromOpenAIForRoute converts an OpenAI chat body into routing signals.
// n>1, store:true and an unsupported service_tier are handler 400s; they do not change
// which model can serve the request. Clearing them here lets route-ahead still rewrite
// the model before the proxy. The original body is what the handler validates.
func fromOpenAIForRoute(oreq openai.ChatCompletionRequest) (*api.ChatRequest, error) {
	if oreq.N != nil && *oreq.N > 1 {
		oreq.N = nil
	}
	if oreq.Store != nil && *oreq.Store {
		oreq.Store = nil
	}
	oreq.ServiceTier = ""
	return openai.FromChatRequest(oreq)
}

// openAIChatNeedsConversion detects fields the native ChatRequest unmarshal would ignore.
func openAIChatNeedsConversion(body []byte, oreq openai.ChatCompletionRequest) bool {
	if len(oreq.Functions) > 0 || oreq.ToolChoice != nil || oreq.FunctionCall != nil {
		return true
	}
	if oreq.MaxTokens != nil || oreq.MaxCompletionTokens != nil {
		return true
	}
	if oreq.PromptCacheKey != nil || oreq.SessionID != nil {
		return true
	}
	// Content-part arrays / image_url / input_audio only exist on the OpenAI Message type.
	var probe struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(body, &probe) != nil {
		return false
	}
	for _, m := range probe.Messages {
		var shape struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(m, &shape) != nil || len(shape.Content) == 0 {
			continue
		}
		trim := bytes.TrimSpace(shape.Content)
		if len(trim) > 0 && trim[0] == '[' {
			return true
		}
	}
	return false
}

// routeInventoryRow is one candidate with its live availability (GET /api/router/inventory).
type routeInventoryRow struct {
	autoroute.Candidate
	Warm          bool    `json:"warm"`
	CanLoad       bool    `json:"can_load"`
	NeedsEviction bool    `json:"needs_eviction"`
	Excluded      string  `json:"excluded,omitempty"`
	Detail        string  `json:"detail,omitempty"`
	TokPerSec     float64 `json:"tok_per_sec,omitempty"`
	GenSec        float64 `json:"gen_sec,omitempty"`
}

// RouterInventoryHandler serves GET /api/router/inventory: what the auto-router can see —
// memoised capabilities, context, bench perf and live residency — plus models that failed
// to load into the inventory. Debug surface; never loads a model.
func (s *Server) RouterInventoryHandler(c *gin.Context) {
	cands, failed, err := routeCandidates()
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	avail := s.routeAvailability(c.Request.Context(), autoroute.Signals{Surface: autoroute.SurfaceChat})
	rows := make([]routeInventoryRow, 0, len(cands))
	for _, cand := range cands {
		a := avail(cand)
		row := routeInventoryRow{
			Candidate: cand, Warm: a.Warm, CanLoad: a.CanLoad, NeedsEviction: a.NeedsEviction,
			Excluded: a.Excluded, Detail: a.Detail,
		}
		if e, ok := cand.Perf(); ok {
			row.TokPerSec, row.GenSec = e.TokPerSec, e.GenSec
		}
		rows = append(rows, row)
	}
	failedOut := map[string]string{}
	for name, ferr := range failed {
		failedOut[name] = ferr.Error()
	}
	body := gin.H{
		"auto_router": envconfig.AutoRouter(),
		"candidates":  rows,
		"failed":      failedOut,
		"routers":     routerSummaries(),
	}
	if _, err := loadRouterFile(); err != nil {
		body["router_config_error"] = err.Error()
	}
	c.JSON(http.StatusOK, body)
}

// routerSummary is the policy view of one router.yaml router: enough for eval tooling
// (scripts/eval/router_eval.py) to score tier / strength accuracy without a YAML parser.
type routerSummary struct {
	Classifier string            `json:"classifier"`
	Decider    string            `json:"decider,omitempty"`
	Fallback   string            `json:"fallback,omitempty"`
	Questions  int               `json:"questions,omitempty"`
	Candidates []RouterCandidate `json:"candidates"`
}

func routerSummaries() map[string]routerSummary {
	out := map[string]routerSummary{}
	file, err := loadRouterFile()
	if err != nil || file == nil {
		return out
	}
	for name, spec := range file.Routers {
		out[name] = routerSummary{
			Classifier: spec.Classifier,
			Decider:    spec.Decider,
			Fallback:   spec.Fallback,
			Questions:  len(spec.Questions),
			Candidates: spec.Candidates,
		}
	}
	return out
}

// autoRouterVersionCapability is advertised under zerollama.capabilities.auto_router on
// GET /api/version so clients probe once before sending model "auto".
func autoRouterVersionCapability() map[string]any {
	return map[string]any{
		"enabled":   envconfig.AutoRouter() && envconfig.RouterRewrite(),
		"name":      implicitAutoRouter,
		"surfaces":  []string{string(autoroute.SurfaceChat), string(autoroute.SurfaceGenerate), string(autoroute.SurfaceImage), string(autoroute.SurfaceImageEdit)},
		"dry_run":   "/v1/route",
		"inventory": "/api/router/inventory",
	}
}

// virtualRouterModelGet answers GET /v1/models/{router} for router names listed on
// /v1/models. WHY: OpenAI clients often validate a model id with a retrieve call before
// using it; ShowHandler 404s on a router name, so "auto" appeared in the list but was
// rejected on retrieve. Real models fall through to the normal retrieve path.
func (s *Server) virtualRouterModelGet() gin.HandlerFunc {
	return func(c *gin.Context) {
		name := strings.TrimSpace(c.Param("model"))
		if c.Request.Method != http.MethodGet || name == "" || !isRouterName(name) {
			c.Next()
			return
		}
		want := model.ParseName(name).DisplayShortest()
		for _, m := range virtualRouterModels(nil) {
			if model.ParseName(m.Name).DisplayShortest() != want {
				continue
			}
			caps := make([]string, len(m.Capabilities))
			for i, cp := range m.Capabilities {
				caps[i] = string(cp)
			}
			c.AbortWithStatusJSON(http.StatusOK, openai.Model{
				Id:           name,
				Object:       "model",
				OwnedBy:      model.ParseName(name).Namespace,
				Capabilities: caps,
			})
			return
		}
		c.Next()
	}
}

// virtualRouterModels lists router names as /v1/models entries so OpenAI-style pickers can
// select "auto". Capabilities are the union over local candidates (implicit "auto") or the
// router's candidate models (router.yaml).
//
// WHY /v1/models only, not /api/tags: zerollama bench / launch / ls enumerate /api/tags and
// would try to benchmark or configure a name that is not a model.
func virtualRouterModels(existing []api.ListModelResponse) []api.ListModelResponse {
	if !envconfig.RouterRewrite() {
		return nil
	}
	have := map[string]bool{}
	for _, m := range existing {
		have[strings.ToLower(m.Name)] = true
	}
	cands, _, err := routeCandidates()
	if err != nil {
		return nil
	}
	byName := map[string]autoroute.Candidate{}
	for _, c := range cands {
		byName[c.Name] = c
	}
	union := func(names []string) []model.Capability {
		seen := map[model.Capability]bool{}
		var out []model.Capability
		for _, n := range names {
			for _, cp := range byName[model.ParseName(n).DisplayShortest()].Capabilities {
				if !seen[cp] {
					seen[cp] = true
					out = append(out, cp)
				}
			}
		}
		return out
	}

	var out []api.ListModelResponse
	add := func(name string, caps []model.Capability) {
		if have[strings.ToLower(name)] {
			return
		}
		have[strings.ToLower(name)] = true
		out = append(out, api.ListModelResponse{
			Name:         name,
			Model:        name,
			Details:      api.ModelDetails{Family: "router"},
			Capabilities: caps,
		})
	}

	if file, err := loadRouterFile(); err == nil && file != nil {
		for name, spec := range file.Routers {
			if len(spec.Candidates) == 0 {
				continue
			}
			names := make([]string, 0, len(spec.Candidates))
			for _, rc := range spec.Candidates {
				names = append(names, rc.Model)
			}
			add(name, union(names))
		}
	}
	// No local candidates ⇒ "auto" could never route; listing it would only produce 503s.
	if len(cands) > 0 && isImplicitAutoName(implicitAutoRouter) {
		names := make([]string, 0, len(cands))
		for _, c := range cands {
			names = append(names, c.Name)
		}
		add(implicitAutoRouter, union(names))
	}
	return out
}
