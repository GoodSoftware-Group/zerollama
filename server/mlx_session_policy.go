package server

import (
	"context"
	"strings"
)

// mlxScheduleHints carries route metadata for server-inferred session class.
type mlxScheduleHints struct {
	Route    string // generate, chat, openai, image_generation, video_generation, unknown
	Modality string // text, vision, video_understanding, image_generation, video_generation
	Stream   bool
}

type mlxScheduleHintsKey struct{}

type mlxSessionMeta struct {
	RawKey     string
	SessionKey string
	Class      mlxSessionClass
	QoS        mlxQoS
}

type mlxSessionMetaKey struct{}

func ctxWithMLXScheduleHints(ctx context.Context, hints mlxScheduleHints) context.Context {
	return context.WithValue(ctx, mlxScheduleHintsKey{}, hints)
}

func mlxScheduleHintsFromCtx(ctx context.Context) mlxScheduleHints {
	if ctx == nil {
		return mlxScheduleHints{Route: "unknown", Stream: true}
	}
	if v, ok := ctx.Value(mlxScheduleHintsKey{}).(mlxScheduleHints); ok {
		return v
	}
	return mlxScheduleHints{Route: "unknown", Stream: true}
}

// classifyMLXSession resolves intent: explicit harness QoS first, then generic inference.
func classifyMLXSession(rawKey string, hints mlxScheduleHints, opts map[string]any) (mlxSessionClass, mlxQoS) {
	qos := mlxQoSFromOptions(opts)
	if qos.Class != mlxClassUnknown {
		return qos.Class, qos
	}

	key := strings.TrimSpace(rawKey)
	switch {
	case strings.HasPrefix(key, "hermes:agent:"):
		return mlxClassInteractive, qos
	case ephemeralSessionKey.MatchString(key):
		return mlxClassAuxiliary, qos
	case strings.HasPrefix(key, "ruby-trivia:bg:"), strings.HasPrefix(key, "bg:"):
		return mlxClassBackground, qos
	case strings.HasPrefix(key, "aux:"):
		return mlxClassAuxiliary, qos
	case key != "":
		// Keyed but no explicit QoS — auxiliary, not interactive peer.
		return mlxClassAuxiliary, qos
	}

	switch hints.Modality {
	case mlxModalityImageGeneration, mlxModalityVideoGeneration:
		return mlxClassBackground, qos
	case mlxModalityVideoUnderstanding, mlxModalityVision:
		if hints.Route == "chat" || hints.Route == "openai" {
			if !hints.Stream {
				return mlxClassBackground, qos
			}
			return mlxClassAuxiliary, qos
		}
	}

	switch hints.Route {
	case "generate", "image_generation", "video_generation":
		return mlxClassBackground, qos
	case "chat", "openai":
		if !hints.Stream {
			return mlxClassBackground, qos
		}
		return mlxClassAuxiliary, qos
	default:
		return mlxClassBackground, qos
	}
}

func sharedBranchPrefix(class mlxSessionClass) string {
	switch class {
	case mlxClassAuxiliary:
		return "aux"
	case mlxClassBackground:
		return "bg"
	default:
		return ""
	}
}

func shouldShareCacheBranch(rawKey string, class mlxSessionClass, qos mlxQoS) bool {
	if qos.CacheScope == qosCacheScopeThread {
		return false
	}
	if qos.CacheScope == qosCacheScopeShared {
		return class == mlxClassAuxiliary || class == mlxClassBackground
	}
	// auto
	key := strings.TrimSpace(rawKey)
	if class == mlxClassInteractive {
		return false
	}
	if key == "" {
		return true
	}
	return ephemeralSessionKey.MatchString(key)
}

// injectMLXSessionKey maps ephemeral/unkeyed traffic onto stable trie branches per model runner.
func injectMLXSessionKey(modelKey, rawKey string, class mlxSessionClass, qos mlxQoS) string {
	key := strings.TrimSpace(rawKey)
	if class == mlxClassInteractive || !shouldShareCacheBranch(key, class, qos) {
		return key
	}
	prefix := sharedBranchPrefix(class)
	if prefix == "" {
		return key
	}
	branch := prefix + ":" + modelKey
	if g := strings.TrimSpace(qos.SessionGroup); g != "" {
		branch += ":" + g
	}
	return branch
}

// mlxDeferPolicy decides whether incoming should wait for the hot slot.
func mlxDeferPolicy(
	incomingClass mlxSessionClass,
	incomingKey string,
	hotClass mlxSessionClass,
	hotKey string,
	inflight int,
) (shouldDefer bool, policy string) {
	if hotKey == "" {
		return false, "idle"
	}
	if incomingClass == mlxClassInteractive {
		if hotClass == mlxClassInteractive {
			if hotKey != incomingKey {
				return true, "interactive_different_thread"
			}
			return false, "interactive_same_thread"
		}
		if inflight > 0 {
			return true, "interactive_wait_inflight_lower"
		}
		return false, "interactive_preempt_cooldown"
	}
	if hotClass == mlxClassInteractive {
		return true, "lower_wait_interactive"
	}
	if incomingClass == mlxClassBackground && hotClass == mlxClassAuxiliary {
		if hotKey == incomingKey {
			return false, "same_branch"
		}
		return true, "background_wait_auxiliary"
	}
	if hotKey == incomingKey {
		return false, "same_branch"
	}
	return true, "different_branch"
}

func prepareMLXSession(ctx context.Context, m *Model, opts map[string]any) (context.Context, mlxSessionMeta) {
	meta := mlxSessionMeta{}
	if m == nil || !m.IsMLX() {
		return ctx, meta
	}
	hints := mlxScheduleHintsFromCtx(ctx)
	meta.RawKey = mlxSessionKey(opts)
	meta.Class, meta.QoS = classifyMLXSession(meta.RawKey, hints, opts)
	modelKey := schedulerModelKey(m)
	meta.SessionKey = injectMLXSessionKey(modelKey, meta.RawKey, meta.Class, meta.QoS)
	if meta.SessionKey != meta.RawKey {
		if opts == nil {
			opts = map[string]any{}
		}
		opts["prompt_cache_key"] = meta.SessionKey
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, mlxSessionMetaKey{}, meta), meta
}

func mlxSessionMetaFromCtx(ctx context.Context) (mlxSessionMeta, bool) {
	if ctx == nil {
		return mlxSessionMeta{}, false
	}
	meta, ok := ctx.Value(mlxSessionMetaKey{}).(mlxSessionMeta)
	return meta, ok
}

// mergeOptionsMaps shallow-merges overlay into base (eliza nested merge preserved).
func mergeOptionsMaps(base, overlay map[string]any) map[string]any {
	if base == nil {
		out := make(map[string]any, len(overlay))
		for k, v := range overlay {
			out[k] = v
		}
		return out
	}
	out := make(map[string]any, len(base)+len(overlay))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overlay {
		if k == "eliza" {
			if existing, ok := out[k].(map[string]any); ok {
				if eliza, ok := v.(map[string]any); ok {
					out[k] = mergeOptionsMaps(existing, eliza)
					continue
				}
			}
		}
		if k == "zerollama" {
			if existing, ok := out[k].(map[string]any); ok {
				if z, ok := v.(map[string]any); ok {
					out[k] = mergeOptionsMaps(existing, z)
					continue
				}
			}
		}
		out[k] = v
	}
	return out
}
