package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// Routing memory (docs/auto-router.md AR6).
//
// Sticky: a conversation keeps its routed model across turns. WHY: switching models
// mid-thread discards the KV / Radix prefix and changes the assistant's voice — for agent
// loops that costs more than a slightly better model would gain. The pick is re-routed
// only when the remembered model fails a hard filter for this turn (images appear, the
// context outgrew it, it was deleted or is cooling down), or the client asks for fresh.
//
// Decision cache: identical text requests within a short window reuse the last pick.
// WHY: agent retries and parallel fan-out would otherwise pay the classifier / Jev
// decider call again for the same prompt.
//
// Both only remember text surfaces; media requests are independent and embed / rerank are
// pinned anyway. Remembered models are hints, always re-validated before reuse.
const (
	routeCacheTTL      = 60 * time.Second
	routeCacheEntries  = 1024
	routeStickyEntries = 4096
)

var (
	routeStickyOnce sync.Once
	routeSticky     *autoroute.Memory
	routeCache      = autoroute.NewMemory(routeCacheTTL, routeCacheEntries)
)

func stickyMemory() *autoroute.Memory {
	routeStickyOnce.Do(func() {
		routeSticky = autoroute.NewMemory(envconfig.RouterStickyTTL(), routeStickyEntries)
	})
	return routeSticky
}

// resolveRouterMem wraps resolveRouterUncached with sticky + cache memory. write is true
// for real traffic; dry runs (/v1/route, /api/router/decide) read memory so they report
// what the next request would do, but never change it.
func (s *Server) resolveRouterMem(ctx context.Context, name string, sig autoroute.Signals, write bool) (*RouterDecision, bool, error) {
	dec, isRouter, err := s.resolveRouterRemembered(ctx, name, sig, write)
	if err == nil && isRouter {
		s.attachSuggestion(ctx, name, sig, dec, write)
	}
	return dec, isRouter, err
}

func (s *Server) resolveRouterRemembered(ctx context.Context, name string, sig autoroute.Signals, write bool) (*RouterDecision, bool, error) {
	if !isRouterName(name) {
		return nil, false, nil
	}
	if !sig.Surface.IsText() {
		return s.resolveRouterUncached(ctx, name, sig)
	}
	start := time.Now()
	stickyKey := ""
	if sig.SessionKey != "" {
		stickyKey = name + "\x00" + sig.SessionKey
	}
	cacheKey := routeCacheKey(name, sig)

	stickyBreak := ""
	if !sig.Fresh {
		lookup := stickyMemory().Peek
		cacheLookup := routeCache.Peek
		if write {
			lookup = stickyMemory().Get
			cacheLookup = routeCache.Get
		}
		if hint, ok := lookup(stickyKey); ok {
			if dec, reason := s.validateRouteHint(ctx, name, sig, hint); dec != nil {
				dec.Sticky, dec.Reason = true, "sticky"
				dec.LatencyMs = time.Since(start).Milliseconds()
				return dec, true, nil
			} else {
				stickyBreak = hint + ": " + reason
				if write {
					stickyMemory().Forget(stickyKey)
				}
			}
		} else if hint, ok := cacheLookup(cacheKey); ok {
			if dec, _ := s.validateRouteHint(ctx, name, sig, hint); dec != nil {
				dec.Cached, dec.Reason = true, "cached"
				dec.LatencyMs = time.Since(start).Milliseconds()
				if write {
					stickyMemory().Put(stickyKey, dec.Candidate)
				}
				return dec, true, nil
			}
		}
	}

	dec, isRouter, err := s.resolveRouterUncached(ctx, name, sig)
	if dec != nil {
		dec.StickyBreak = stickyBreak
	}
	if err == nil && dec != nil && write {
		stickyMemory().Put(stickyKey, dec.Candidate)
		routeCache.Put(cacheKey, dec.Candidate)
	}
	return dec, isRouter, err
}

// validateRouteHint re-checks a remembered model for this request. It returns a decision
// when the model can still serve, else the reason it cannot.
func (s *Server) validateRouteHint(ctx context.Context, name string, sig autoroute.Signals, hint string) (*RouterDecision, string) {
	if !routerAllowsModel(name, hint) {
		return nil, "no longer a candidate of this router"
	}
	cands, _, err := routeCandidates()
	if err != nil {
		return nil, err.Error()
	}
	want := model.ParseName(hint).DisplayShortest()
	for _, c := range cands {
		// Normalize untagged router.yaml hints (coder) against inventory names (coder:latest).
		if model.ParseName(c.Name).DisplayShortest() != want {
			continue
		}
		if reason, ok := autoroute.Validate(c, sig, s.routeAvailability(ctx, sig)); !ok {
			return nil, reason
		}
		return &RouterDecision{Router: name, Classifier: routerClassifier(name), Candidate: c.Name}, ""
	}
	return nil, "model is no longer local"
}

// routerAllowsModel reports whether model is still one of router name's choices — a
// router.yaml edit may have removed it. The implicit router may pick any local model.
func routerAllowsModel(name, m string) bool {
	spec, ok := lookupRouter(name)
	if !ok {
		return true
	}
	want := model.ParseName(m).DisplayShortest()
	match := func(n string) bool {
		return strings.TrimSpace(n) != "" && model.ParseName(n).DisplayShortest() == want
	}
	for _, c := range spec.Candidates {
		if match(c.Model) {
			return true
		}
	}
	return match(spec.Fallback) || match(spec.Embedder) || match(spec.Reranker)
}

func routerClassifier(name string) string {
	if spec, ok := lookupRouter(name); ok && spec.Classifier != "" {
		return spec.Classifier
	}
	return "deterministic"
}

// routeCacheKey identifies an identical text request for the decision cache: router,
// surface, every capability-relevant flag, a coarse token bucket and the exact text.
func routeCacheKey(name string, sig autoroute.Signals) string {
	if strings.TrimSpace(sig.Text) == "" {
		return ""
	}
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s\x00%t%t%t%t%t\x00%d\x00%d\x00", name, sig.Surface,
		sig.HasImages, sig.HasVideo, sig.HasAudio, sig.HasTools, sig.WantsThink,
		sig.PromptTokens/512, sig.ReplyTokens)
	h.Write([]byte(sig.Text))
	return hex.EncodeToString(h.Sum(nil))
}

// routeSessionKey derives a conversation identity: the client's cache / session key when
// present, else a hash of the system prompt and first user message — both stay the same on
// every turn of one conversation.
func routeSessionKey(opts map[string]any, msgs []api.Message) string {
	if k := modality.ExtractPromptCacheKey(opts); k != "" {
		return "key:" + k
	}
	var system, first strings.Builder
	for _, m := range msgs {
		switch {
		case strings.EqualFold(m.Role, "system"):
			system.WriteString(m.Content)
			system.WriteByte('\n')
		case strings.EqualFold(m.Role, "user") && first.Len() == 0:
			first.WriteString(m.Content)
		}
	}
	if first.Len() == 0 {
		return ""
	}
	sum := sha256.Sum256([]byte(system.String() + "\x00" + first.String()))
	return "conv:" + hex.EncodeToString(sum[:12])
}

// routeFresh reports options.zerollama.route == "fresh" — skip remembered picks this turn.
func routeFresh(opts map[string]any) bool {
	z, ok := opts["zerollama"].(map[string]any)
	if !ok {
		return false
	}
	v, _ := z["route"].(string)
	return strings.EqualFold(strings.TrimSpace(v), "fresh")
}

// resetRouteMemory clears sticky + cache memory and re-reads the sticky TTL. Tests call it
// so routing memory never leaks between cases (the tables are process-global).
func resetRouteMemory() {
	routeStickyOnce = sync.Once{}
	routeSticky = nil
	routeCache = autoroute.NewMemory(routeCacheTTL, routeCacheEntries)
}
