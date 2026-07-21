package server

import (
	"context"
	"log/slog"
)

// Package-level note: QoS scheduling for agent harnesses.
//
// WHY reserveScheduleQoS exists: MLX live KV is keyed by prompt_cache_key. Concurrent
// streams that pass defer checks but claim the runner later can race switchToPath and
// kill the subprocess (Jul 2026 production). We claim the gate slot before GetRunner.
//
// WHY GGUF branching differs from MLX: GGUF/llama-server L3 uses the client key verbatim;
// rewriting aux/bg keys onto shared branches (MLX trie policy) would desync ps labels
// and cache_n. Unkeyed GGUF traffic is a no-op — CUDA batch endpoints must not wait
// behind MLX agent cooldown they do not participate in.
//
// WHY fulfillment: complete/benchmark are request-scoped no-degradation contracts
// (SQL-transaction-like begin→release). They force interactive class, inject a gate
// key when omitted, and for benchmark unload peer models for exclusive GPU speed.
//
// WHY ParentKey is passed into waitForSlot: multiplex wait_parent checks the key
// hot-map (and inject candidates), not only the fairness primary — otherwise a child
// waits forever behind the wrong agent or never waits when parent is still hot.
//
// See docs/agent-qos-and-project-tracking.md.

// scheduleSessionMeta resolves gate session key + class for MLX and GGUF runners.
func scheduleSessionMeta(ctx context.Context, m *Model, opts map[string]any) (sessionKey string, class mlxSessionClass, qos mlxQoS) {
	if m == nil {
		return "", mlxClassUnknown, mlxQoS{}
	}
	hints := mlxScheduleHintsFromCtx(ctx)
	if opts == nil {
		opts = map[string]any{}
	}
	ensureQoSDefaults(opts, hints)
	qos = mlxQoSFromOptions(opts)

	if m.IsMLX() {
		_, meta := prepareMLXSession(ctx, m, opts)
		sessionKey = meta.SessionKey
		if sessionKey == "" {
			sessionKey = mlxSessionKey(opts)
		}
		class = meta.Class
		if class == mlxClassUnknown {
			class, qos = classifyMLXSession(meta.RawKey, hints, opts)
		} else {
			qos = meta.QoS
		}
		if qos.Fulfillment.Active() {
			sessionKey = ensureFulfillmentSessionKey(opts, qos)
			class = mlxClassInteractive
		}
		return sessionKey, class, qos
	}

	rawKey := ensureFulfillmentSessionKey(opts, qos)
	class, qos = classifyMLXSession(rawKey, hints, opts)
	if qos.Fulfillment.Active() {
		class = mlxClassInteractive
		rawKey = ensureFulfillmentSessionKey(opts, qos)
	}
	modelKey := schedulerModelKey(m)
	sessionKey = gateSessionKey(m, modelKey, rawKey, class, qos)
	return sessionKey, class, qos
}

// waitScheduleQoS blocks until session policy allows scheduling. MLX and GGUF text
// share the per-model gate; non-interactive work also waits behind any hot interactive slot.
func (s *Server) waitScheduleQoS(ctx context.Context, m *Model, opts map[string]any) error {
	_, err := s.reserveScheduleQoS(ctx, m, opts)
	return err
}

// reserveScheduleQoS waits for session policy, then immediately claims the gate slot
// so concurrent requests cannot pass defer checks while this one waits for a runner.
func (s *Server) reserveScheduleQoS(ctx context.Context, m *Model, opts map[string]any) (func(), error) {
	if s == nil || s.sched == nil || m == nil {
		return func() {}, nil
	}
	sessionKey, class, qos := scheduleSessionMeta(ctx, m, opts)
	modelKey := schedulerModelKey(m)

	// Fulfillment opts into the gate even when modelSupportsSessionQoS would skip,
	// as long as we have a session key (injected if needed).
	if !qos.Fulfillment.Active() && !modelSupportsSessionQoS(m) {
		return func() {}, nil
	}

	if sessionKey == "" {
		// No session key: MLX unkeyed traffic waits behind any hot interactive slot
		// (preserves KV trie for agent sessions). Non-MLX runners are unaffected —
		// they don't participate in the trie and shouldn't be delayed by MLX policy.
		if !m.IsMLX() {
			return func() {}, nil
		}
		if err := s.sched.mlxGate.waitBehindAnyInteractive(ctx, class, injectMLXSessionKey(gpuMediaModelKey, "", class, mlxQoS{})); err != nil {
			return func() {}, err
		}
		return func() {}, nil
	}

	if qos.Fulfillment.Active() {
		if err := s.sched.mlxGate.waitForFulfillment(ctx, modelKey, sessionKey, qos.Fulfillment); err != nil {
			return func() {}, err
		}
	}
	if err := s.sched.mlxGate.waitForSlot(ctx, modelKey, sessionKey, qos.ParentKey, class, qos); err != nil {
		return func() {}, err
	}
	if err := s.sched.mlxGate.waitBehindAnyInteractive(ctx, class, sessionKey); err != nil {
		return func() {}, err
	}

	releaseSession := s.sched.mlxGate.begin(modelKey, sessionKey, class, qos)
	releaseFulfill := s.sched.mlxGate.beginFulfillment(modelKey, sessionKey, qos.Fulfillment)

	if qos.Fulfillment.Exclusive() {
		// Benchmark mode: free peer VRAM so the speed path is not degraded by
		// concurrent resident models. Keep this model if already loaded.
		s.sched.unloadRunnersExcept(modelKey)
		slog.Info("fulfillment benchmark: unloaded peer runners",
			"model_key", modelKey,
			"session_key", sessionKey,
		)
	}

	return func() {
		releaseFulfill()
		releaseSession()
	}, nil
}

// agentSessionBegin is deprecated; reserveScheduleQoS claims the gate before runner wait.
func (s *Server) agentSessionBegin(ctx context.Context, m *Model, opts map[string]any, inferencing bool) func() {
	return func() {}
}
