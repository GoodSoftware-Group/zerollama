package server

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/agentstats"
	"github.com/ollama/ollama/api"
)

const (
	// mlxSidecarAgentCooldown is how long after a keyed turn completes the
	// runner stays "hot" for that session key. Covers the ~20–30s gap
	// between rapid Discord turns where a competing session would otherwise
	// switchToPath and destroy live KV.
	mlxSidecarAgentCooldown = 90 * time.Second
)

// mlxSessionSlot tracks a single keyed session for one MLX runner (model key).
type mlxSessionSlot struct {
	sessionKey   string
	sessionClass mlxSessionClass
	sessionGroup string
	projectID    string
	projectName  string
	inflight     int
	hotUntil     time.Time
}

// mlxAgentGate serialises competing prompt_cache_key sessions on each model runner
// so that switchToPath cannot clobber live KV between agent turns.
//
// WHY not MLX-only despite the name: GGUF text with explicit session keys also
// registers here for interactive/background defer and zerollama ps metadata.
// Unkeyed GGUF is excluded at reserveScheduleQoS (see inference_path.go).
//
// Intent-aware policy (practical ladder step 1–2):
//   - interactive (hermes:agent:*, explicit qos): holds slot; never defers behind lower cooldown
//   - auxiliary (ephemeral spawns): defers behind primary; shares aux:{model} branch
//   - background (unkeyed /api/generate, ruby-trivia:bg:*): lowest priority
//   - fulfillment complete/benchmark: request-scoped no-degradation / exclusive holds
//   - pins: session TTL leases that block eviction without loading (Phase B3).
//     Soft UnloadAllRunners respects them; Forced (training/bench) does not.
//     RuntimeGGUFs soft-pin Python residency in Go (503 on conflicting GGUF).
type mlxAgentGate struct {
	mu      sync.Mutex
	slots   map[string]*mlxSessionSlot
	fulfill *fulfillmentHold
	pins    map[string]*pinLease // pin_id → lease
}

func newMLXAgentGate() *mlxAgentGate {
	return &mlxAgentGate{
		slots: make(map[string]*mlxSessionSlot),
		pins:  make(map[string]*pinLease),
	}
}

func (g *mlxAgentGate) begin(modelKey, sessionKey string, class mlxSessionClass, qos mlxQoS) func() {
	if modelKey == "" || sessionKey == "" {
		return func() {}
	}
	g.mu.Lock()
	slot := g.slots[modelKey]
	if slot == nil {
		slot = &mlxSessionSlot{}
		g.slots[modelKey] = slot
	}
	slot.sessionKey = sessionKey
	slot.sessionClass = class
	if qos.SessionGroup != "" {
		slot.sessionGroup = qos.SessionGroup
	}
	if qos.ProjectID != "" {
		slot.projectID = qos.ProjectID
	}
	if qos.ProjectName != "" {
		slot.projectName = qos.ProjectName
	}
	slot.inflight++
	g.mu.Unlock()
	return func() { g.end(modelKey, sessionKey) }
}

// activeSessionsForModel returns hot or in-flight gate sessions for GET /api/ps.
func (g *mlxAgentGate) activeSessionsForModel(modelKey string, now time.Time) []api.ProcessSessionInfo {
	if g == nil || modelKey == "" {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	slot := g.slots[modelKey]
	if slot == nil || slot.sessionKey == "" {
		return nil
	}
	if slot.inflight <= 0 && !now.Before(slot.hotUntil) {
		return nil
	}
	return []api.ProcessSessionInfo{slot.processSessionInfo(now)}
}

// activeSessionsSnapshot returns all hot/in-flight sessions across loaded models.
func (g *mlxAgentGate) activeSessionsSnapshot(now time.Time) map[string][]api.ProcessSessionInfo {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make(map[string][]api.ProcessSessionInfo)
	for modelKey, slot := range g.slots {
		if slot == nil || slot.sessionKey == "" {
			continue
		}
		if slot.inflight <= 0 && !now.Before(slot.hotUntil) {
			continue
		}
		out[modelKey] = []api.ProcessSessionInfo{slot.processSessionInfo(now)}
	}
	return out
}

func (s *mlxSessionSlot) processSessionInfo(now time.Time) api.ProcessSessionInfo {
	info := api.ProcessSessionInfo{
		SessionKey:   s.sessionKey,
		SessionClass: s.sessionClass.String(),
		SessionGroup: s.sessionGroup,
		ProjectID:    s.projectID,
		ProjectName:  s.projectName,
		Inflight:     s.inflight,
	}
	if s.inflight <= 0 && now.Before(s.hotUntil) {
		info.HotUntil = s.hotUntil
	}
	return info
}

func formatProcessProjectLabel(id, name string) string {
	id = strings.TrimSpace(id)
	name = strings.TrimSpace(name)
	switch {
	case id != "" && name != "" && !strings.EqualFold(id, name):
		return id + "/" + name
	case name != "":
		return name
	case id != "":
		return id
	default:
		return ""
	}
}

func (g *mlxAgentGate) end(modelKey, sessionKey string) {
	if modelKey == "" || sessionKey == "" {
		return
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	slot := g.slots[modelKey]
	if slot == nil || slot.sessionKey != sessionKey {
		return
	}
	slot.inflight--
	if slot.inflight <= 0 {
		slot.inflight = 0
		slot.hotUntil = now.Add(mlxSidecarAgentCooldown)
	}
}

func (g *mlxAgentGate) hotSlot(modelKey string, now time.Time) (sessionKey string, class mlxSessionClass, inflight int) {
	slot := g.slots[modelKey]
	if slot == nil || slot.sessionKey == "" {
		return "", mlxClassUnknown, 0
	}
	if slot.inflight > 0 || now.Before(slot.hotUntil) {
		return slot.sessionKey, slot.sessionClass, slot.inflight
	}
	return "", mlxClassUnknown, 0
}

func (g *mlxAgentGate) shouldDefer(
	modelKey string,
	incomingKey string,
	incomingClass mlxSessionClass,
	now time.Time,
) (bool, string, string) {
	if deferF, policy := g.shouldDeferFulfillment(modelKey, incomingKey, incomingClass); deferF {
		hold, _ := g.fulfillmentActive(now)
		return true, policy, hold.sessionKey
	}
	hotKey, hotClass, inflight := g.hotSlot(modelKey, now)
	defer_, policy := mlxDeferPolicy(incomingClass, incomingKey, hotClass, hotKey, inflight)
	return defer_, policy, hotKey
}

func (g *mlxAgentGate) waitForSlot(
	ctx context.Context,
	modelKey string,
	incomingKey string,
	incomingClass mlxSessionClass,
) error {
	if modelKey == "" {
		return nil
	}
	start := time.Now()
	var blockedByKey string
	var blockedByClass mlxSessionClass
	var lastPolicy string
	for {
		g.mu.Lock()
		now := time.Now()
		defer_, policy, hotKey := g.shouldDefer(modelKey, incomingKey, incomingClass, now)
		if defer_ && blockedByKey == "" {
			blockedByKey = hotKey
			_, blockedByClass, _ = g.hotSlot(modelKey, now)
			lastPolicy = policy
		}
		g.mu.Unlock()

		if !defer_ {
			if waited := time.Since(start); waited >= 100*time.Millisecond {
				slog.Info("mlx session deferred",
					"model_key", modelKey,
					"incoming_key", incomingKey,
					"incoming_class", incomingClass.String(),
					"blocked_by", blockedByKey,
					"blocked_by_class", blockedByClass.String(),
					"policy", lastPolicy,
					"waited", waited,
				)
				recordMLXSidecarDefer("deferred", modelKey, incomingKey, incomingClass, blockedByKey, blockedByClass, lastPolicy, waited, "")
			}
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func (g *mlxAgentGate) hottestInteractive(now time.Time) (hotKey string, hotClass mlxSessionClass, inflight int, hotModelKey string) {
	for mk, slot := range g.slots {
		if slot == nil || slot.sessionKey == "" || slot.sessionClass != mlxClassInteractive {
			continue
		}
		if slot.inflight > 0 || now.Before(slot.hotUntil) {
			return slot.sessionKey, slot.sessionClass, slot.inflight, mk
		}
	}
	return "", mlxClassUnknown, 0, ""
}

// waitBehindAnyInteractive blocks non-interactive GPU media work while any MLX interactive
// session is hot (Wan video, external imagegen, etc. are not MLX runners). Exclusive
// fulfillment holds also block other interactive traffic.
func (g *mlxAgentGate) waitBehindAnyInteractive(ctx context.Context, incomingClass mlxSessionClass, incomingKey string) error {
	start := time.Now()
	var blockedByKey, hotModelKey string
	var blockedByClass mlxSessionClass
	var lastPolicy string
	for {
		g.mu.Lock()
		now := time.Now()
		deferNow := false
		if deferF, policy := g.shouldDeferFulfillment("", incomingKey, incomingClass); deferF {
			hold, _ := g.fulfillmentActive(now)
			// complete holds only force non-interactive media behind them globally;
			// exclusive benchmark holds block everyone including other interactive.
			if hold.mode.Exclusive() || incomingClass != mlxClassInteractive {
				deferNow = true
				if blockedByKey == "" {
					blockedByKey = hold.sessionKey
					blockedByClass = mlxClassInteractive
					hotModelKey = hold.modelKey
					lastPolicy = policy
				}
			}
		}
		if !deferNow && incomingClass != mlxClassInteractive {
			hotKey, hotClass, inflight, mk := g.hottestInteractive(now)
			defer_, policy := mlxDeferPolicy(incomingClass, incomingKey, hotClass, hotKey, inflight)
			if defer_ {
				deferNow = true
				if blockedByKey == "" {
					blockedByKey = hotKey
					blockedByClass = hotClass
					hotModelKey = mk
					lastPolicy = policy
				}
			}
		}
		g.mu.Unlock()

		if !deferNow {
			if waited := time.Since(start); waited >= 100*time.Millisecond && lastPolicy != "" {
				slog.Info("mlx global defer",
					"incoming_key", incomingKey,
					"incoming_class", incomingClass.String(),
					"blocked_by", blockedByKey,
					"blocked_by_model", hotModelKey,
					"blocked_by_class", blockedByClass.String(),
					"policy", lastPolicy,
					"waited", waited,
				)
				recordMLXSidecarDefer("global_deferred", hotModelKey, incomingKey, incomingClass, blockedByKey, blockedByClass, lastPolicy, waited, "media_qos")
			}
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func (g *mlxAgentGate) mlxNeedsDefer(
	m *Model,
	modelKey string,
	incomingKey string,
	incomingClass mlxSessionClass,
) (bool, string) {
	if m == nil || !m.IsMLX() {
		return false, ""
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	defer_, policy, _ := g.shouldDefer(modelKey, incomingKey, incomingClass, time.Now())
	return defer_, policy
}

// mlxSessionKey extracts the prompt_cache_key from request options.
func mlxSessionKey(opts map[string]any) string {
	return strings.TrimSpace(promptCacheKeyFromOptions(opts))
}

func (s *Server) mlxAgentSessionBegin(ctx context.Context, m *Model, opts map[string]any, inferencing bool) func() {
	return s.agentSessionBegin(ctx, m, opts, inferencing)
}

func (s *Server) waitMLXSessionIdle(ctx context.Context, m *Model, opts map[string]any) error {
	return s.waitScheduleQoS(ctx, m, opts)
}

func recordMLXSidecarDefer(
	action, modelKey, incomingKey string,
	incomingClass mlxSessionClass,
	hotKey string,
	hotClass mlxSessionClass,
	policy string,
	waited time.Duration,
	reason string,
) {
	fields := map[string]any{
		"event":          "mlx_sidecar_defer",
		"action":         action,
		"model_key":      modelKey,
		"incoming_key":   incomingKey,
		"incoming_class": incomingClass.String(),
		"hot_key":        hotKey,
		"hot_class":      hotClass.String(),
		"policy":         policy,
		"waited_ms":      waited.Milliseconds(),
	}
	if reason != "" {
		fields["reason"] = reason
	}
	agentstats.Record("mlx_sidecar_defer", fields)
}
