package server

import (
	"context"
	"log/slog"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// fulfillmentHold tracks an in-flight complete/benchmark reservation.
// WHY not a long lease: this is request-scoped (begin → release), like a SQL
// transaction — not a multi-minute fleet quote.
type fulfillmentHold struct {
	mode       fulfillmentMode
	modelKey   string
	sessionKey string
}

const (
	// fulfillCompleteKeepAliveFloor keeps the runner warm across bench/eval think gaps
	// without claiming exclusive GPU forever.
	fulfillCompleteKeepAliveFloor = 30 * time.Minute
	// fulfillBenchmarkKeepAliveFloor keeps the pinned model resident for multi-epoch benches.
	fulfillBenchmarkKeepAliveFloor = 2 * time.Hour
)

// ensureFulfillmentSessionKey injects a stable gate key when the client omitted
// prompt_cache_key. Fulfillment must participate in the session gate even on
// unkeyed GGUF (normally a QoS no-op).
func ensureFulfillmentSessionKey(opts map[string]any, qos mlxQoS) string {
	if !qos.Fulfillment.Active() {
		return mlxSessionKey(opts)
	}
	if key := mlxSessionKey(opts); key != "" {
		return key
	}
	project := strings.TrimSpace(qos.ProjectID)
	if project == "" {
		project = "anon"
	}
	key := "fulfill:" + qos.Fulfillment.String() + ":" + project
	if opts == nil {
		return key
	}
	opts["prompt_cache_key"] = key
	return key
}

// fulfillmentKeepAliveFloor raises keep_alive when unset so unload cannot degrade
// an in-progress fulfillment request. Explicit keep_alive (including 0) is honored.
func fulfillmentKeepAliveFloor(qos mlxQoS, ka *api.Duration) *api.Duration {
	if !qos.Fulfillment.Active() {
		return ka
	}
	if ka != nil {
		return ka
	}
	floor := envconfig.KeepAlive()
	want := fulfillCompleteKeepAliveFloor
	if qos.Fulfillment.Exclusive() {
		want = fulfillBenchmarkKeepAliveFloor
	}
	if floor < want {
		floor = want
	}
	d := api.Duration{Duration: floor}
	return &d
}

func (g *mlxAgentGate) fulfillmentActive(now time.Time) (hold fulfillmentHold, ok bool) {
	if g == nil {
		return fulfillmentHold{}, false
	}
	if g.fulfill == nil {
		return fulfillmentHold{}, false
	}
	hold = *g.fulfill
	if !hold.mode.Active() || hold.sessionKey == "" {
		return fulfillmentHold{}, false
	}
	return hold, true
}

func (g *mlxAgentGate) beginFulfillment(modelKey, sessionKey string, mode fulfillmentMode) func() {
	if g == nil || !mode.Active() || modelKey == "" || sessionKey == "" {
		return func() {}
	}
	g.mu.Lock()
	g.fulfill = &fulfillmentHold{
		mode:       mode,
		modelKey:   modelKey,
		sessionKey: sessionKey,
	}
	g.mu.Unlock()
	slog.Info("fulfillment hold begin",
		"mode", mode.String(),
		"exclusive", mode.Exclusive(),
		"model_key", modelKey,
		"session_key", sessionKey,
	)
	return func() {
		g.mu.Lock()
		if g.fulfill != nil &&
			g.fulfill.sessionKey == sessionKey &&
			g.fulfill.modelKey == modelKey {
			g.fulfill = nil
		}
		g.mu.Unlock()
		slog.Info("fulfillment hold end",
			"mode", mode.String(),
			"model_key", modelKey,
			"session_key", sessionKey,
		)
	}
}

// waitForFulfillment clears the path for a complete/benchmark request.
// complete: wait until this model has no competing hot holder (or same key).
// benchmark: wait until no other fulfillment hold and no other inflight sessions.
func (g *mlxAgentGate) waitForFulfillment(
	ctx context.Context,
	modelKey, sessionKey string,
	mode fulfillmentMode,
) error {
	if g == nil || !mode.Active() {
		return nil
	}
	start := time.Now()
	for {
		g.mu.Lock()
		now := time.Now()
		ready, reason := g.fulfillmentReadyLocked(modelKey, sessionKey, mode, now)
		g.mu.Unlock()
		if ready {
			if waited := time.Since(start); waited >= 100*time.Millisecond {
				slog.Info("fulfillment wait cleared",
					"mode", mode.String(),
					"model_key", modelKey,
					"session_key", sessionKey,
					"waited", waited,
					"reason", reason,
				)
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

func (g *mlxAgentGate) fulfillmentReadyLocked(
	modelKey, sessionKey string,
	mode fulfillmentMode,
	now time.Time,
) (ready bool, reason string) {
	if hold, ok := g.fulfillmentActive(now); ok {
		if hold.sessionKey == sessionKey && hold.modelKey == modelKey {
			return true, "same_fulfillment"
		}
		return false, "other_fulfillment:" + hold.mode.String()
	}
	if mode.Exclusive() {
		for mk, slot := range g.slots {
			if slot == nil || slot.sessionKey == "" {
				continue
			}
			if slot.sessionKey == sessionKey && mk == modelKey {
				continue
			}
			if slot.inflight > 0 {
				return false, "inflight:" + slot.sessionKey
			}
			if now.Before(slot.hotUntil) && slot.sessionClass == mlxClassInteractive {
				return false, "hot_interactive:" + slot.sessionKey
			}
		}
		return true, "exclusive_idle"
	}
	// complete: only need this model clear (or same thread).
	hotKey, hotClass, inflight := g.hotSlot(modelKey, now)
	if hotKey == "" {
		return true, "model_idle"
	}
	if hotKey == sessionKey {
		return true, "same_thread"
	}
	if hotClass == mlxClassInteractive || inflight > 0 {
		return false, "model_busy:" + hotKey
	}
	return true, "preempt_cooldown"
}

// shouldDeferFulfillment blocks other traffic while a fulfillment hold is active.
func (g *mlxAgentGate) shouldDeferFulfillment(
	incomingModelKey, incomingKey string,
	incomingClass mlxSessionClass,
) (bool, string) {
	hold, ok := g.fulfillmentActive(time.Now())
	if !ok {
		return false, ""
	}
	if hold.sessionKey == incomingKey && hold.modelKey == incomingModelKey {
		return false, "same_fulfillment"
	}
	if hold.mode.Exclusive() {
		return true, "fulfillment_exclusive"
	}
	// complete: protect the held model; allow unrelated interactive on other models.
	if hold.modelKey == incomingModelKey {
		return true, "fulfillment_complete_same_model"
	}
	if incomingClass != mlxClassInteractive {
		return true, "fulfillment_complete_lower"
	}
	return false, ""
}

// protectedModelKeys returns model keys that must not be eviction victims.
func (g *mlxAgentGate) protectedModelKeys() map[string]struct{} {
	out := make(map[string]struct{})
	if g == nil {
		return out
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if hold, ok := g.fulfillmentActive(time.Now()); ok {
		out[hold.modelKey] = struct{}{}
	}
	return out
}
