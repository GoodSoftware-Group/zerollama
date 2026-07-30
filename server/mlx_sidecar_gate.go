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

	// mlxKeyHotCap bounds per-model session keys tracked for multiplexed
	// wait_parent (many agents on one connection).
	mlxKeyHotCap = 64
)

// mlxSessionSlot tracks the primary keyed session for one runner (model key).
// Interactive fairness still uses this single primary; multiplex parent waits
// use the per-key hot map alongside it.
type mlxSessionSlot struct {
	sessionKey   string
	sessionClass mlxSessionClass
	sessionGroup string
	parentKey    string
	projectID    string
	projectName  string
	cacheScope   string
	cacheLevel   string
	fulfillment  string
	inflight     int
	hotUntil     time.Time
}

// mlxKeyHotEntry tracks one prompt_cache_key's inflight/cooldown on a model.
type mlxKeyHotEntry struct {
	sessionKey   string
	sessionClass mlxSessionClass
	sessionGroup string
	parentKey    string
	projectID    string
	projectName  string
	cacheScope   string
	cacheLevel   string
	fulfillment  string
	inflight     int
	hotUntil     time.Time
	lastUsed     time.Time
	// Soft mid-stream preempt (M15f): cancel inflight lower-class decode when
	// interactive admits. One-shot; cleared after fire / takePreemptReason.
	preemptCancel context.CancelFunc
	preemptReason string
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
//   - session_parent: wait_parent when parent key is inflight/cooldown (multiplex-aware
//     key hot-map — WHY: primary alone fails when another agent claimed last)
//   - pins: session TTL leases that block eviction without loading (Phase B3).
//     Soft UnloadAllRunners respects them; Forced (training/bench) does not.
//     RuntimeGGUFs soft-pin Python residency in Go (503 on conflicting GGUF).
//
// WHY primary is derived from keyHot: begin used to ++slot.inflight and overwrite
// sessionKey; end only decremented on key match → concurrent claims leaked hot
// fairness. keyHot is the source of truth; slots is the fairness/ps projection.
type mlxAgentGate struct {
	mu      sync.Mutex
	slots   map[string]*mlxSessionSlot
	keyHot  map[string]map[string]*mlxKeyHotEntry // modelKey → sessionKey → entry
	fulfill *fulfillmentHold
	pins    map[string]*pinLease // pin_id → lease
}

func newMLXAgentGate() *mlxAgentGate {
	return &mlxAgentGate{
		slots:  make(map[string]*mlxSessionSlot),
		keyHot: make(map[string]map[string]*mlxKeyHotEntry),
		pins:   make(map[string]*pinLease),
	}
}

func applyQoSToSlot(slot *mlxSessionSlot, sessionKey string, class mlxSessionClass, qos mlxQoS) {
	slot.sessionKey = sessionKey
	slot.sessionClass = class
	if qos.SessionGroup != "" {
		slot.sessionGroup = qos.SessionGroup
	}
	if qos.ParentKey != "" {
		slot.parentKey = qos.ParentKey
	}
	if qos.ProjectID != "" {
		slot.projectID = qos.ProjectID
	}
	if qos.ProjectName != "" {
		slot.projectName = qos.ProjectName
	}
	if qos.CacheScope != "" {
		slot.cacheScope = qos.CacheScope
	}
	if qos.CacheLevel != "" {
		slot.cacheLevel = qos.CacheLevel
	}
	if qos.Fulfillment.Active() {
		slot.fulfillment = qos.Fulfillment.String()
	}
}

func (g *mlxAgentGate) begin(modelKey, sessionKey string, class mlxSessionClass, qos mlxQoS) func() {
	if modelKey == "" || sessionKey == "" {
		return func() {}
	}
	now := time.Now()
	g.mu.Lock()
	g.beginLocked(modelKey, sessionKey, class, qos, now)
	g.mu.Unlock()
	return func() { g.end(modelKey, sessionKey) }
}

// beginLocked claims a session key under g.mu. Primary-slot display/hotness is
// derived from the key hot-map so concurrent claims cannot leak inflight.
func (g *mlxAgentGate) beginLocked(modelKey, sessionKey string, class mlxSessionClass, qos mlxQoS, now time.Time) {
	if g.slots == nil {
		g.slots = make(map[string]*mlxSessionSlot)
	}
	if g.keyHot == nil {
		g.keyHot = make(map[string]map[string]*mlxKeyHotEntry)
	}
	if g.slots[modelKey] == nil {
		g.slots[modelKey] = &mlxSessionSlot{}
	}
	g.upsertKeyHotLocked(modelKey, sessionKey, class, qos, now, +1)
	g.refreshPrimaryFromKeyHotLocked(modelKey, now)
}

func (g *mlxAgentGate) upsertKeyHotLocked(modelKey, sessionKey string, class mlxSessionClass, qos mlxQoS, now time.Time, deltaInflight int) {
	m := g.keyHot[modelKey]
	if m == nil {
		m = make(map[string]*mlxKeyHotEntry)
		g.keyHot[modelKey] = m
	}
	entry := m[sessionKey]
	if entry == nil {
		entry = &mlxKeyHotEntry{sessionKey: sessionKey}
		m[sessionKey] = entry
	}
	entry.sessionClass = class
	if qos.SessionGroup != "" {
		entry.sessionGroup = qos.SessionGroup
	}
	if qos.ParentKey != "" {
		entry.parentKey = qos.ParentKey
	}
	if qos.ProjectID != "" {
		entry.projectID = qos.ProjectID
	}
	if qos.ProjectName != "" {
		entry.projectName = qos.ProjectName
	}
	if qos.CacheScope != "" {
		entry.cacheScope = qos.CacheScope
	}
	if qos.CacheLevel != "" {
		entry.cacheLevel = qos.CacheLevel
	}
	if qos.Fulfillment.Active() {
		entry.fulfillment = qos.Fulfillment.String()
	}
	entry.inflight += deltaInflight
	if entry.inflight < 0 {
		entry.inflight = 0
	}
	entry.lastUsed = now
	g.evictKeyHotLocked(modelKey, now)
}

// refreshPrimaryFromKeyHotLocked picks the fairness "primary" from keyHot:
// interactive inflight > interactive cooldown > any inflight > any cooldown.
func (g *mlxAgentGate) refreshPrimaryFromKeyHotLocked(modelKey string, now time.Time) {
	slot := g.slots[modelKey]
	if slot == nil {
		slot = &mlxSessionSlot{}
		g.slots[modelKey] = slot
	}
	m := g.keyHot[modelKey]
	var best *mlxKeyHotEntry
	bestScore := -1
	totalInflight := 0
	for _, e := range m {
		if e == nil {
			continue
		}
		if e.inflight > 0 {
			totalInflight += e.inflight
		}
		if !e.isHot(now) {
			continue
		}
		score := 0
		if e.sessionClass == mlxClassInteractive {
			score += 4
		}
		if e.inflight > 0 {
			score += 2
		} else {
			score += 1
		}
		if score > bestScore || (score == bestScore && best != nil && e.lastUsed.After(best.lastUsed)) {
			best = e
			bestScore = score
		}
	}
	if best == nil {
		slot.inflight = 0
		slot.hotUntil = time.Time{}
		return
	}
	slot.sessionKey = best.sessionKey
	slot.sessionClass = best.sessionClass
	slot.sessionGroup = best.sessionGroup
	slot.parentKey = best.parentKey
	slot.projectID = best.projectID
	slot.projectName = best.projectName
	slot.cacheScope = best.cacheScope
	slot.cacheLevel = best.cacheLevel
	slot.fulfillment = best.fulfillment
	slot.inflight = totalInflight
	if best.inflight > 0 {
		slot.hotUntil = time.Time{}
	} else {
		slot.hotUntil = best.hotUntil
	}
}

func (g *mlxAgentGate) evictKeyHotLocked(modelKey string, now time.Time) {
	m := g.keyHot[modelKey]
	if m == nil {
		return
	}
	// Drop expired cooldown entries first.
	for k, e := range m {
		if e.inflight > 0 || now.Before(e.hotUntil) {
			continue
		}
		delete(m, k)
	}
	for len(m) > mlxKeyHotCap {
		var oldestKey string
		var oldest time.Time
		for k, e := range m {
			if e.inflight > 0 {
				continue
			}
			if oldestKey == "" || e.lastUsed.Before(oldest) {
				oldestKey = k
				oldest = e.lastUsed
			}
		}
		if oldestKey == "" {
			break
		}
		delete(m, oldestKey)
	}
}

func (e *mlxKeyHotEntry) isHot(now time.Time) bool {
	return e != nil && (e.inflight > 0 || now.Before(e.hotUntil))
}

// parentKeyCandidates expands session_parent to gate keys that may have been
// rewritten via injectMLXSessionKey (aux:/bg: shared branches).
//
// WHY: children usually send the raw parent prompt_cache_key; MLX may have
// claimed the parent under aux:{model}[:group]. Exact match alone misses wait_parent.
func parentKeyCandidates(modelKey, parentKey string, qos mlxQoS) []string {
	key := strings.TrimSpace(parentKey)
	if key == "" {
		return nil
	}
	seen := map[string]struct{}{key: {}}
	out := []string{key}
	add := func(s string) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}
		if _, ok := seen[s]; ok {
			return
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	for _, class := range []mlxSessionClass{mlxClassInteractive, mlxClassAuxiliary, mlxClassBackground} {
		add(injectMLXSessionKey(modelKey, key, class, qos))
		if qos.SessionGroup != "" {
			add(injectMLXSessionKey(modelKey, key, class, mlxQoS{}))
		}
	}
	return out
}

func (g *mlxAgentGate) parentHotLocked(modelKey, parentKey string, qos mlxQoS, now time.Time) (bool, string) {
	parentKey = strings.TrimSpace(parentKey)
	if parentKey == "" || modelKey == "" {
		return false, ""
	}
	m := g.keyHot[modelKey]
	if m == nil {
		return false, ""
	}
	for _, candidate := range parentKeyCandidates(modelKey, parentKey, qos) {
		if m[candidate].isHot(now) {
			return true, candidate
		}
	}
	return false, ""
}

// activeSessionsForModel returns hot or in-flight gate sessions for GET /api/ps.
func (g *mlxAgentGate) activeSessionsForModel(modelKey string, now time.Time) []api.ProcessSessionInfo {
	if g == nil || modelKey == "" {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.sessionsForModelLocked(modelKey, now)
}

// activeSessionsSnapshot returns all hot/in-flight sessions across loaded models.
func (g *mlxAgentGate) activeSessionsSnapshot(now time.Time) map[string][]api.ProcessSessionInfo {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make(map[string][]api.ProcessSessionInfo)
	for modelKey := range g.slots {
		if sessions := g.sessionsForModelLocked(modelKey, now); len(sessions) > 0 {
			out[modelKey] = sessions
		}
	}
	// Include models that only have key-hot entries (primary cleared).
	for modelKey := range g.keyHot {
		if _, ok := out[modelKey]; ok {
			continue
		}
		if sessions := g.sessionsForModelLocked(modelKey, now); len(sessions) > 0 {
			out[modelKey] = sessions
		}
	}
	return out
}

func (g *mlxAgentGate) sessionsForModelLocked(modelKey string, now time.Time) []api.ProcessSessionInfo {
	seen := make(map[string]struct{})
	var out []api.ProcessSessionInfo

	if slot := g.slots[modelKey]; slot != nil && slot.sessionKey != "" {
		if slot.inflight > 0 || now.Before(slot.hotUntil) {
			out = append(out, slot.processSessionInfo(now))
			seen[slot.sessionKey] = struct{}{}
		}
	}
	for key, entry := range g.keyHot[modelKey] {
		if entry == nil || !entry.isHot(now) {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		out = append(out, entry.processSessionInfo(now))
		seen[key] = struct{}{}
	}
	return out
}

func (s *mlxSessionSlot) processSessionInfo(now time.Time) api.ProcessSessionInfo {
	info := api.ProcessSessionInfo{
		SessionKey:    s.sessionKey,
		SessionClass:  s.sessionClass.String(),
		SessionGroup:  s.sessionGroup,
		SessionParent: s.parentKey,
		ProjectID:     s.projectID,
		ProjectName:   s.projectName,
		CacheScope:    s.cacheScope,
		CacheLevel:    s.cacheLevel,
		Fulfillment:   s.fulfillment,
		Inflight:      s.inflight,
	}
	if s.inflight <= 0 && now.Before(s.hotUntil) {
		info.HotUntil = s.hotUntil
	}
	return info
}

func (e *mlxKeyHotEntry) processSessionInfo(now time.Time) api.ProcessSessionInfo {
	info := api.ProcessSessionInfo{
		SessionKey:    e.sessionKey,
		SessionClass:  e.sessionClass.String(),
		SessionGroup:  e.sessionGroup,
		SessionParent: e.parentKey,
		ProjectID:     e.projectID,
		ProjectName:   e.projectName,
		CacheScope:    e.cacheScope,
		CacheLevel:    e.cacheLevel,
		Fulfillment:   e.fulfillment,
		Inflight:      e.inflight,
	}
	if e.inflight <= 0 && now.Before(e.hotUntil) {
		info.HotUntil = e.hotUntil
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
	if m := g.keyHot[modelKey]; m != nil {
		if entry := m[sessionKey]; entry != nil {
			entry.inflight--
			if entry.inflight <= 0 {
				entry.inflight = 0
				entry.hotUntil = now.Add(mlxSidecarAgentCooldown)
			}
			entry.lastUsed = now
		}
	}
	// Primary fairness state is derived from keyHot — never match-and-decrement
	// slot.sessionKey (that leaked inflight when a later begin overwrote the key).
	g.refreshPrimaryFromKeyHotLocked(modelKey, now)
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
	parentKey string,
	incomingClass mlxSessionClass,
	qos mlxQoS,
	now time.Time,
) (bool, string, string) {
	if deferF, policy := g.shouldDeferFulfillment(modelKey, incomingKey, incomingClass); deferF {
		hold, _ := g.fulfillmentActive(now)
		return true, policy, hold.sessionKey
	}
	if hot, matched := g.parentHotLocked(modelKey, parentKey, qos, now); hot {
		return true, "wait_parent", matched
	}
	hotKey, hotClass, inflight := g.hotSlot(modelKey, now)
	defer_, policy := mlxDeferPolicy(incomingClass, incomingKey, hotClass, hotKey, inflight)
	return defer_, policy, hotKey
}

func (g *mlxAgentGate) waitForSlot(
	ctx context.Context,
	modelKey string,
	incomingKey string,
	parentKey string,
	incomingClass mlxSessionClass,
	qos mlxQoS,
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
		defer_, policy, hotKey := g.shouldDefer(modelKey, incomingKey, parentKey, incomingClass, qos, now)
		if defer_ && blockedByKey == "" {
			blockedByKey = hotKey
			_, blockedByClass, _ = g.hotSlot(modelKey, now)
			if policy == "wait_parent" {
				blockedByClass = mlxClassInteractive
			}
			lastPolicy = policy
		}
		// Soft-preempt lower-class inflight so interactive need not wait out a full decode.
		// WHY under same lock: cancelSessionLocked reads keyHot; already hold g.mu.
		if defer_ && policy == "interactive_wait_inflight_lower" && hotKey != "" {
			g.cancelSessionLocked(modelKey, hotKey, "lower_wait_interactive")
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
			return wrapQoSDeferAbort(ctx.Err(), lastPolicy)
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
			return wrapQoSDeferAbort(ctx.Err(), lastPolicy)
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
	defer_, policy, _ := g.shouldDefer(modelKey, incomingKey, "", incomingClass, mlxQoS{}, time.Now())
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
