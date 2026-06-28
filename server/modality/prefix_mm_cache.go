// SGLang enable_prefix_mm_cache compatibility.
//
// WHY a separate file: SGLang clients set enable_prefix_mm_cache when they expect
// per-conversation ViT encoder reuse. Zerollama already pins session ViT overlay
// when prompt_cache_key is set (same key as L3 + session video/layout caches).
// The flag documents operator intent; without a session key we log once per request
// so misconfigured agents are visible in fleet logs — overlay stays off either way.
package modality

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
)

// OptionsEnablePrefixMMCache reports SGLang enable_prefix_mm_cache on request options.
func OptionsEnablePrefixMMCache(opts map[string]any) bool {
	if opts == nil {
		return false
	}
	return optionsBool(opts, "enable_prefix_mm_cache")
}

func optionsBool(opts map[string]any, key string) bool {
	v, ok := opts[key]
	if !ok {
		return false
	}
	switch t := v.(type) {
	case bool:
		return t
	case string:
		s := strings.TrimSpace(strings.ToLower(t))
		return s == "1" || s == "true" || s == "yes"
	default:
		return false
	}
}

// SessionViTOverlayEnabled reports whether per-thread ViT embed pinning is active.
//
// WHY: SGLang enable_prefix_mm_cache gates MultiModalStaticCache lookup. Zerollama
// defaults overlay ON when prompt_cache_key is set (agent turn-2 speed); set
// enable_prefix_mm_cache=false to disable session pin and rely on global LRU only.
func SessionViTOverlayEnabled(opts map[string]any) bool {
	key := ExtractPromptCacheKey(opts)
	if key == "" {
		return false
	}
	if opts == nil {
		return true
	}
	if _, ok := opts["enable_prefix_mm_cache"]; !ok {
		return true
	}
	return optionsBool(opts, "enable_prefix_mm_cache")
}
// Session ViT overlay and video/layout session caches require prompt_cache_key (or eliza aliases).
func WarnPrefixMMCacheWithoutSessionKey(req *api.ChatRequest) {
	if req == nil || !OptionsEnablePrefixMMCache(req.Options) {
		return
	}
	if ExtractPromptCacheKey(req.Options) != "" {
		return
	}
	slog.Info("enable_prefix_mm_cache without prompt_cache_key; session ViT/layout overlay disabled",
		"hint", "set prompt_cache_key or options.eliza.promptCacheKey for agent-thread pinning",
	)
}
