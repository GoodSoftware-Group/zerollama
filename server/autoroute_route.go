package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// implicitAutoRouter is the reserved router name served without router.yaml (AR1).
const implicitAutoRouter = "auto"

// isImplicitAutoName reports whether name should use the built-in deterministic router.
// A router.yaml entry of the same name wins (checked by the caller first), and so does a
// real local model tagged "auto" — operators who already have one keep it.
func isImplicitAutoName(name string) bool {
	if !envconfig.AutoRouter() {
		return false
	}
	n := model.ParseName(strings.TrimSpace(name))
	if !n.IsValid() || !strings.EqualFold(n.Model, implicitAutoRouter) || !strings.EqualFold(n.Tag, "latest") ||
		!strings.EqualFold(n.Namespace, model.DefaultName().Namespace) || !strings.EqualFold(n.Host, model.DefaultName().Host) {
		return false
	}
	return !localManifestExists(n)
}

func localManifestExists(n model.Name) bool {
	ms, err := manifest.Manifests(true)
	if err != nil {
		return false
	}
	for e := range ms {
		if strings.EqualFold(e.Host, n.Host) && strings.EqualFold(e.Namespace, n.Namespace) &&
			strings.EqualFold(e.Model, n.Model) && strings.EqualFold(e.Tag, n.Tag) {
			return true
		}
	}
	return false
}

// decideAuto runs the deterministic router over every local model. On error the returned
// decision is non-nil (Auto.Rejected explains the filtering) unless listing models failed.
//
// allow restricts candidates to those names (router.yaml candidates on non-text surfaces);
// nil means every local model.
func (s *Server) decideAuto(ctx context.Context, router string, sig autoroute.Signals, allow map[string]bool) (*RouterDecision, error) {
	start := time.Now()
	cands, failed, err := routeCandidates()
	if err != nil {
		return nil, fmt.Errorf("router %q: list models: %w", router, err)
	}
	for name, ferr := range failed {
		slog.Debug("auto-router: skipping unreadable model", "model", name, "error", ferr)
	}
	if allow != nil {
		kept := cands[:0:0]
		for _, c := range cands {
			if allow[c.Name] {
				kept = append(kept, c)
			}
		}
		cands = kept
	}
	var helperRejections []autoroute.Rejection
	if allow == nil && sig.Surface.IsText() {
		cands, helperRejections = withoutRouterHelpers(cands)
	}
	d, err := autoroute.Decide(cands, sig, s.routeAvailability(ctx, sig))
	d.Rejected = append(helperRejections, d.Rejected...)
	dec := &RouterDecision{
		Router:     router,
		Classifier: "deterministic",
		Candidate:  d.Model,
		Reason:     d.Reason,
		Auto:       &d,
		LatencyMs:  time.Since(start).Milliseconds(),
	}
	if err != nil {
		// dec still carries Auto.Rejected so dry runs can list why each model was dropped.
		return dec, fmt.Errorf("router %q: %w", router, err)
	}
	return dec, nil
}

// withoutRouterHelpers drops models that router.yaml routers use internally (score
// classifier, jev decider, embedder, reranker) from implicit "auto" text candidates.
//
// WHY: found on real models (docs/auto-router-findings.md Finding 12) — the warm bonus made
// implicit auto send every prompt to the resident 1B score classifier, because it was the
// only model loaded. Helpers stay eligible when a router also lists them as a candidate,
// and the filter is soft: a node whose only models are helpers still gets an answer.
func withoutRouterHelpers(cands []autoroute.Candidate) ([]autoroute.Candidate, []autoroute.Rejection) {
	helpers, serving := routerHelperModels()
	if len(helpers) == 0 {
		return cands, nil
	}
	var kept []autoroute.Candidate
	var rejected []autoroute.Rejection
	for _, c := range cands {
		if role, ok := helpers[c.Name]; ok && !serving[c.Name] {
			rejected = append(rejected, autoroute.Rejection{Model: c.Name, Reason: "router helper (" + role + "), not an implicit auto pick"})
			continue
		}
		kept = append(kept, c)
	}
	if len(kept) == 0 {
		return cands, nil
	}
	return kept, rejected
}

// routerHelperModels returns models router.yaml uses as helpers (name → role) and the set
// of models any router lists as a serving candidate or fallback.
func routerHelperModels() (map[string]string, map[string]bool) {
	file, err := loadRouterFile()
	if err != nil || file == nil {
		return nil, nil
	}
	norm := func(n string) string { return model.ParseName(strings.TrimSpace(n)).DisplayShortest() }
	helpers, serving := map[string]string{}, map[string]bool{}
	add := func(n, role string) {
		if strings.TrimSpace(n) != "" {
			helpers[norm(n)] = role
		}
	}
	for _, spec := range file.Routers {
		switch strings.ToLower(strings.TrimSpace(spec.Classifier)) {
		case "", "knn", "rerank", "colbert", "jev":
		default:
			add(spec.Classifier, "score classifier")
		}
		add(spec.Decider, "jev decider")
		add(spec.Embedder, "embedder")
		add(spec.Reranker, "reranker")
		for _, c := range spec.Candidates {
			serving[norm(c.Model)] = true
		}
		if strings.TrimSpace(spec.Fallback) != "" {
			serving[norm(spec.Fallback)] = true
		}
	}
	return helpers, serving
}

// enforceRouterSurface checks the model an LA11 classifier picked against the request's
// surface and capabilities, swapping in the router's fallback when only that qualifies.
//
// WHY: LA11 candidates are label-matched and surface-blind — a router name sent to
// /v1/images/* could resolve to a chat model, and a tools request to a model without tools.
// Models missing from the local inventory (cloud, not pulled) pass through unchanged so
// the handler reports them exactly as before.
func enforceRouterSurface(dec *RouterDecision, spec RouterSpec, sig autoroute.Signals) error {
	cands, _, err := routeCandidates()
	if err != nil {
		return nil
	}
	byName := make(map[string]autoroute.Candidate, len(cands))
	for _, c := range cands {
		byName[c.Name] = c
	}
	admits := func(name string) (string, bool) {
		c, ok := byName[model.ParseName(name).DisplayShortest()]
		if !ok {
			return "", true
		}
		return sig.Admits(c)
	}
	reason, ok := admits(dec.Candidate)
	if ok {
		return nil
	}
	if fb := spec.Fallback; fb != "" && fb != dec.Candidate {
		if _, fbOK := admits(fb); fbOK {
			dec.Candidate = fb
			dec.Fallback = true
			dec.Reason = "surface-fallback"
			return nil
		}
	}
	return fmt.Errorf("router %q chose %q, which cannot serve this %s request (%s)", dec.Router, dec.Candidate, sig.Surface, reason)
}

// routeAvailability snapshots scheduler residency once per routed request and returns a
// per-candidate dry-run probe. It applies the same rules as evaluateCanLoadGgml (warm,
// concurrency group, max loaded) plus load cooldown, without its runtime-health probe and
// GPU discovery, which would otherwise repeat for every candidate. It never loads.
func (s *Server) routeAvailability(ctx context.Context, sig autoroute.Signals) autoroute.AvailabilityFunc {
	runtimeOn := effectiveRuntimeURL() != ""
	loaded := map[string]bool{}
	loadsPaused := false
	var sched *Scheduler
	if s != nil && s.sched != nil {
		sched = s.sched
		loadsPaused = sched.loadsPaused.Load()
		sched.loadedMu.Lock()
		for key, r := range sched.loaded {
			if r != nil {
				loaded[key] = true
			}
		}
		sched.loadedMu.Unlock()
	}
	// WHY lazy: effectiveMaxLoadedModels may run GPU discovery (tens of seconds on a cold
	// cache). With nothing resident no load can evict, so only ask once residents exist.
	maxLoaded := sync.OnceValue(func() uint { return effectiveMaxLoadedModels(ctx) })

	return func(c autoroute.Candidate) autoroute.Availability {
		// WHY excluded on in-handler routing: the runtime proxy middlewares already ran, so a
		// routed runtime model would hit the ggml path and 400 (ErrRuntimeInferenceModel).
		// routeAheadMiddleware routes /api/chat, /api/generate and /v1/chat/completions
		// before the proxies (BeforeRuntimeProxy), where runtime models are fine.
		if runtimeOn && c.RuntimeInference && sig.Surface.IsText() && !sig.BeforeRuntimeProxy {
			return autoroute.Availability{Excluded: "runtime-sidecar model: auto-routable only on /api/chat, /api/generate, /v1/chat/completions"}
		}
		a := autoroute.Availability{Warm: c.SchedKey != "" && loaded[c.SchedKey], CanLoad: true}
		if a.Warm || sched == nil {
			return a
		}
		if loadsPaused {
			a.CanLoad, a.Detail = false, "loads paused"
			return a
		}
		if err := sched.cooldownErr(c.SchedKey); err != nil {
			a.CanLoad, a.Detail = false, err.Error()
			return a
		}
		if m, err := GetModel(c.ID); err == nil && sched.findConcurrencyGroupConflict(m) != nil {
			a.NeedsEviction, a.Detail = true, "concurrency_group"
			return a
		}
		if len(loaded) > 0 && maxLoaded() > 0 && uint(len(loaded)) >= maxLoaded() {
			a.NeedsEviction, a.Detail = true, "max_loaded_models"
		}
		return a
	}
}

// estimateTokens is a cheap byte-based estimate (~4 bytes/token). WHY not the tokenizer:
// routing runs before any model is chosen, and the context filter only needs the order of
// magnitude — it is soft and never rejects every candidate.
func estimateTokens(bytes int) int { return bytes / 4 }

func chatRouteSignals(c *gin.Context, req *api.ChatRequest) autoroute.Signals {
	sig := autoroute.Signals{
		Surface:            autoroute.SurfaceChat,
		Text:               lastUserMessageText(req.Messages),
		HasVideo:           modality.ChatRequestHasVideoPayload(req),
		HasTools:           len(req.Tools) > 0,
		BeforeRuntimeProxy: true, // live chat routes ahead of the runtime proxy
	}
	n := 0
	for _, msg := range req.Messages {
		if len(msg.Images) > 0 || len(msg.PaddedInputIDs) > 0 {
			sig.HasImages = true
		}
		if len(msg.AudioClips) > 0 {
			sig.HasAudio = true
		}
		n += len(msg.Content) + len(msg.Thinking)
	}
	if sig.HasTools {
		if b, err := json.Marshal(req.Tools); err == nil {
			n += len(b)
		}
	}
	sig.PromptTokens = estimateTokens(n)
	if np, ok := numPredictFromOptions(req.Options); ok {
		sig.ReplyTokens = np
	}
	// Mirror ChatHandler: alias-derived think and Anthropic relax_thinking are no-ops on
	// non-thinking models, so they must not exclude them.
	_, relax := c.Get("relax_thinking")
	sig.WantsThink = req.Think != nil && req.Think.Bool() && !req.ThinkFromAlias && !relax
	sig.SessionKey = routeSessionKey(req.Options, req.Messages)
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if strings.EqualFold(req.Messages[i].Role, "assistant") && strings.TrimSpace(req.Messages[i].Content) != "" {
			sig.LastAssistant = req.Messages[i].Content
			break
		}
	}
	sig.Fresh = routeFresh(req.Options)
	return sig
}

func generateRouteSignals(c *gin.Context, req *api.GenerateRequest) autoroute.Signals {
	sig := autoroute.Signals{Surface: autoroute.SurfaceGenerate, Text: req.Prompt}
	switch c.GetString(middleware.CtxKeyRouteSurface) {
	case string(autoroute.SurfaceImage):
		sig.Surface = autoroute.SurfaceImage
	case string(autoroute.SurfaceImageEdit):
		sig.Surface = autoroute.SurfaceImageEdit
	default:
		// Native /api/generate image requests carry width/height (x/imagegen contract).
		if req.Width > 0 || req.Height > 0 {
			sig.Surface = autoroute.SurfaceImage
		}
	}
	sig.HasImages = len(req.Images) > 0
	sig.PromptTokens = estimateTokens(len(req.Prompt) + len(req.System))
	if np, ok := numPredictFromOptions(req.Options); ok {
		sig.ReplyTokens = np
	}
	sig.WantsThink = req.Think != nil && req.Think.Bool()
	// Generate has no conversation: sticky only with an explicit cache / session key.
	if k := modality.ExtractPromptCacheKey(req.Options); k != "" {
		sig.SessionKey = "key:" + k
	}
	sig.Fresh = routeFresh(req.Options)
	// Live /api/generate text routes ahead of the runtime proxy; image surfaces do not.
	if sig.Surface == autoroute.SurfaceGenerate {
		sig.BeforeRuntimeProxy = true
	}
	return sig
}
