package server

import (
	"context"
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
// See docs/agent-qos-and-project-tracking.md.

// scheduleSessionMeta resolves gate session key + class for MLX and GGUF runners.
func scheduleSessionMeta(ctx context.Context, m *Model, opts map[string]any) (sessionKey string, class mlxSessionClass) {
	if m == nil {
		return "", mlxClassUnknown
	}
	hints := mlxScheduleHintsFromCtx(ctx)
	if opts == nil {
		opts = map[string]any{}
	}
	ensureQoSDefaults(opts, hints)

	if m.IsMLX() {
		_, meta := prepareMLXSession(ctx, m, opts)
		sessionKey = meta.SessionKey
		if sessionKey == "" {
			sessionKey = mlxSessionKey(opts)
		}
		class = meta.Class
		if class == mlxClassUnknown {
			class, _ = classifyMLXSession(meta.RawKey, hints, opts)
		}
		return sessionKey, class
	}

	rawKey := mlxSessionKey(opts)
	class, qos := classifyMLXSession(rawKey, hints, opts)
	modelKey := schedulerModelKey(m)
	sessionKey = gateSessionKey(m, modelKey, rawKey, class, qos)
	return sessionKey, class
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
	if s == nil || s.sched == nil || m == nil || !modelSupportsSessionQoS(m) {
		return func() {}, nil
	}
	sessionKey, class := scheduleSessionMeta(ctx, m, opts)
	modelKey := schedulerModelKey(m)
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
	if err := s.sched.mlxGate.waitForSlot(ctx, modelKey, sessionKey, class); err != nil {
		return func() {}, err
	}
	if err := s.sched.mlxGate.waitBehindAnyInteractive(ctx, class, sessionKey); err != nil {
		return func() {}, err
	}
	qos := mlxQoSFromOptions(opts)
	return s.sched.mlxGate.begin(modelKey, sessionKey, class, qos), nil
}

// agentSessionBegin is deprecated; reserveScheduleQoS claims the gate before runner wait.
func (s *Server) agentSessionBegin(ctx context.Context, m *Model, opts map[string]any, inferencing bool) func() {
	return func() {}
}
