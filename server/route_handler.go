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
}

// routeSignalsFromBody builds Signals from a chat- or generate-shaped body. Surface is
// inferred when empty: messages ⇒ chat; width/height ⇒ image; otherwise generate.
func routeSignalsFromBody(c *gin.Context, body []byte, surface autoroute.Surface) (autoroute.Signals, error) {
	if surface != "" && !routeSurfaces[surface] {
		return autoroute.Signals{}, fmt.Errorf("unsupported surface %q (chat, generate, image, image_edit)", surface)
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
		return chatRouteSignals(c, req), nil
	}

	var req api.GenerateRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return autoroute.Signals{}, err
	}
	if surface == autoroute.SurfaceImage || surface == autoroute.SurfaceImageEdit {
		c.Set(middleware.CtxKeyRouteSurface, string(surface))
	}
	return generateRouteSignals(c, &req), nil
}

// parseRouteChat accepts native /api/chat bodies and OpenAI chat-completions bodies
// (content-part arrays, image_url, input_audio) so signals match what the handler sees.
func parseRouteChat(body []byte) (*api.ChatRequest, error) {
	var native api.ChatRequest
	nativeErr := json.Unmarshal(body, &native)
	if nativeErr == nil {
		return &native, nil
	}
	var oreq openai.ChatCompletionRequest
	if err := json.Unmarshal(body, &oreq); err != nil {
		return nil, errors.Join(nativeErr, err)
	}
	req, err := openai.FromChatRequest(oreq)
	if err != nil {
		return nil, err
	}
	return req, nil
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
