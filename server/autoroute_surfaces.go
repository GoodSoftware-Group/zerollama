package server

import (
	"bytes"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/server/media"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// isRouterName reports whether name resolves through a router (router.yaml or implicit auto).
func isRouterName(name string) bool {
	if _, ok := lookupRouter(name); ok {
		return true
	}
	return isImplicitAutoName(name)
}

// routeAheadMiddleware routes router names before the runtime / SGLang proxy middlewares,
// rewriting the body's "model" so every later stage (proxy or handler) sees a concrete tag.
//
// WHY: those proxies pick a backend from the model name before any handler runs. Routing
// only inside the handler meant a routed runtime-sidecar model landed on the ggml path, so
// AR1 had to exclude runtime models. Routing here makes them first-class on these routes.
// surface "" infers from the body (messages ⇒ chat; width/height ⇒ image; else generate).
func (s *Server) routeAheadMiddleware(surface autoroute.Surface) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost || !envconfig.RouterRewrite() {
			c.Next()
			return
		}
		body, err := readRequestBody(c.Request)
		if err != nil {
			c.Next()
			return
		}
		name, ok := extractModelField(body)
		if !ok {
			c.Next()
			return
		}
		served, aliased, err := resolveAlias(name)
		if err != nil || !isRouterName(served) {
			// Ordinary model (or bad alias): the handler applies the alias and reports errors.
			c.Next()
			return
		}
		sig, err := routeSignalsFromBody(c, body, surface)
		if err != nil {
			c.Next() // malformed body: let the handler produce its usual 400
			return
		}
		sig.BeforeRuntimeProxy = true
		dec, _, err := s.resolveRouterMem(c.Request.Context(), served, sig, true)
		if err != nil {
			metricsRouteError(served, sig.Surface)
			writeRouterError(c, dec, err)
			return
		}
		metricsRouteDecision(dec, sig.Surface)
		rewritten, err := replaceJSONModelField(body, dec.Candidate)
		if err != nil {
			c.Next()
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(rewritten))
		c.Request.ContentLength = int64(len(rewritten))
		if aliased {
			c.Header("X-Zerollama-Alias", name)
			c.Header("X-Zerollama-Alias-Target", served)
		}
		stampRouterHeaders(c, dec)
		c.Next()
	}
}

// routeError is a failed in-handler routing decision; dec may carry filtered candidates.
type routeError struct {
	dec *RouterDecision
	err error
}

func (e *routeError) Error() string { return e.err.Error() }
func (e *routeError) Unwrap() error { return e.err }

// routeForSurface resolves an (already alias-resolved) model name for a non-chat surface
// and stamps router headers. Non-router names pass through. Callers write errors with
// writeRouterError(c, re.dec, re.err) so every surface reports 503 + filtered the same way.
func (s *Server) routeForSurface(c *gin.Context, name string, sig autoroute.Signals) (string, *routeError) {
	if !envconfig.RouterRewrite() {
		return name, nil
	}
	dec, isRouter, err := s.resolveRouterMem(c.Request.Context(), name, sig, true)
	if !isRouter {
		return name, nil
	}
	if err != nil {
		metricsRouteError(name, sig.Surface)
		return "", &routeError{dec: dec, err: err}
	}
	metricsRouteDecision(dec, sig.Surface)
	stampRouterHeaders(c, dec)
	return dec.Candidate, nil
}

func videoRouteSignals(req openai.VideoCreateRequest) autoroute.Signals {
	_, labels, labelErr := media.ParseKeyframeRefs(req.Options)
	return autoroute.Signals{
		Surface: autoroute.SurfaceVideoGen,
		Text:    req.Prompt,
		// Same checks VideoCreateHandler applies after resolving the model, so a routed
		// request never 400s on backend / keyframes / platform or 503s on host RAM.
		Check: func(c autoroute.Candidate) string {
			m, err := GetModel(c.ID)
			if err != nil {
				return err.Error()
			}
			backend := modality.BackendFor(m.Config, model.ModalityVideoGeneration)
			if err := videoBackendErr(backend); err != nil {
				return err.Error()
			}
			cfg, err := resolveVideoGenerationConfig(m, req)
			if err != nil {
				return err.Error()
			}
			if labelErr == nil {
				if err := validateVideoPlan(backend, cfg, req.Options, labels); err != nil {
					return err.Error()
				}
			}
			if err := admitVideoHostRAM(backend, cfg); err != nil {
				return "host RAM: " + err.Error()
			}
			return ""
		},
	}
}

func speechRouteSignals(req openai.SpeechCreateRequest, surface autoroute.Surface) autoroute.Signals {
	return autoroute.Signals{Surface: surface, Text: req.Input, Voice: req.Voice}
}

// decisionsRouteSignals picks a decider deterministically (docs/auto-router.md §6.11):
// images need a multimodal score encoding (clef); calibrated deciders rank above
// uncalibrated ones; Laya's trained boolean head is preferred for noul questions.
func decisionsRouteSignals(req api.DecisionsRequest) autoroute.Signals {
	hasNoul := false
	for _, q := range req.Questions {
		if q.Type == "noul" {
			hasNoul = true
		}
	}
	text, _ := req.State.(string)
	sig := autoroute.Signals{Surface: autoroute.SurfaceDecisions, Text: text, HasImages: len(req.Images) > 0}
	sig.Check = func(c autoroute.Candidate) string {
		if !sig.HasImages {
			return ""
		}
		m, err := GetModel(c.ID)
		if err != nil {
			return err.Error()
		}
		if !modelUsesSystemOneScore(m) || decisionScoreEncoding(m) != "clef" {
			return "images need a multimodal (clef) decision model"
		}
		return ""
	}
	sig.Bonus = func(c autoroute.Candidate) (float64, string) {
		m, err := GetModel(c.ID)
		if err != nil {
			return 0, ""
		}
		switch {
		case isLayaDecisionModel(m):
			if hasNoul {
				return 0.75, "calibrated (laya noul head)"
			}
			return 0.5, "calibrated"
		case modelUsesSystemOneScore(m), c.Backend(model.ModalityDecisions) == model.BackendCLM:
			return 0.5, "calibrated"
		case c.Backend(model.ModalityDecisions) == model.BackendOpenJev && envconfig.OpenJevCalibrated():
			return 0.25, "calibrated (openjev opt-in)"
		}
		return 0, ""
	}
	return sig
}
