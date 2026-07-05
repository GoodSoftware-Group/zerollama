package server

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// EnsureAgentPromptCacheKey normalizes agent session metadata for L3 slot pinning,
// ViT session overlay, Darwin runtime routing, and MLX trie snapshots.
//
// WHY gating eliza enrichment: plain prompt_cache_key (e.g. openwebui:user:1) is enough
// for llama-server cache_n. Injecting eliza/prefixHash for every key adds unknown fields
// to third-party OpenAI-compatible servers. agentSessionMetadataEnabled limits enrichment
// to options.zerollama, existing eliza blocks, or known harness prefixes.
func EnsureAgentPromptCacheKey(req *api.ChatRequest) {
	if req == nil {
		return
	}
	req.Options = normalizePromptCacheOptions(req.Options, "", req.Messages)
}

// EnsureGeneratePromptCacheKey mirrors chat normalization for /api/generate agent loops.
func EnsureGeneratePromptCacheKey(req *api.GenerateRequest) {
	if req == nil {
		return
	}
	req.Options = normalizePromptCacheOptions(req.Options, req.System, nil)
}

func normalizePromptCacheOptions(opts map[string]any, system string, messages []api.Message) map[string]any {
	key := promptCacheKeyFromOptions(opts)
	if key == "" {
		return opts
	}
	if opts == nil {
		opts = map[string]any{}
	}
	opts["prompt_cache_key"] = key
	if !agentSessionMetadataEnabled(opts) {
		return opts
	}
	backfillElizaPromptCache(opts, key)
	enrichPrefixHash(opts, system, messages)
	return opts
}

func promptCacheKeyFromOptions(opts map[string]any) string {
	if opts != nil {
		if v, ok := opts["prompt_cache_key"].(string); ok && strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return agentPromptCacheKeyFromEliza(opts)
}

func enrichPrefixHash(opts map[string]any, system string, messages []api.Message) {
	h := stableSystemPrefixHash(messages)
	if h == "" && strings.TrimSpace(system) != "" {
		sum := sha256.Sum256([]byte(strings.TrimSpace(system)))
		h = "sys:" + hex.EncodeToString(sum[:8])
	}
	if h == "" {
		return
	}
	ensureElizaBlock(opts)
	eliza := opts["eliza"].(map[string]any)
	if v, ok := eliza["prefixHash"].(string); ok && strings.TrimSpace(v) != "" {
		return
	}
	eliza["prefixHash"] = h
}

func backfillElizaPromptCache(opts map[string]any, key string) {
	ensureElizaBlock(opts)
	eliza := opts["eliza"].(map[string]any)
	if v, ok := eliza["promptCacheKey"].(string); !ok || strings.TrimSpace(v) == "" {
		eliza["promptCacheKey"] = key
	}
	if v, ok := eliza["conversationId"].(string); !ok || strings.TrimSpace(v) == "" {
		if cid := conversationIDFromCacheKey(key); cid != "" {
			eliza["conversationId"] = cid
		}
	}
}

func conversationIDFromCacheKey(key string) string {
	key = strings.TrimSpace(key)
	if after, ok := strings.CutPrefix(key, "hermes:"); ok && after != "" {
		return after
	}
	if after, ok := strings.CutPrefix(key, "conv:"); ok && after != "" {
		return after
	}
	return key
}

func ensureElizaBlock(opts map[string]any) {
	if opts == nil {
		return
	}
	if eliza, ok := opts["eliza"].(map[string]any); ok && eliza != nil {
		return
	}
	opts["eliza"] = map[string]any{}
}

func agentPromptCacheKeyFromEliza(opts map[string]any) string {
	if opts == nil {
		return ""
	}
	eliza, ok := opts["eliza"].(map[string]any)
	if !ok {
		return ""
	}
	if v, ok := eliza["promptCacheKey"].(string); ok && strings.TrimSpace(v) != "" {
		return strings.TrimSpace(v)
	}
	if v, ok := eliza["conversationId"].(string); ok && strings.TrimSpace(v) != "" {
		return "conv:" + strings.TrimSpace(v)
	}
	if v, ok := eliza["conversation_id"].(string); ok && strings.TrimSpace(v) != "" {
		return "conv:" + strings.TrimSpace(v)
	}
	return ""
}

// agentCachePrefersRuntime is true when a stable session key is present and
// operator policy allows routing GGUF agent traffic through the Python runtime
// for L3 prefix cache (Darwin default ggml otherwise skips runtime).
func agentCachePrefersRuntime(opts map[string]any) bool {
	if !envconfig.AgentCacheRuntimeEnabled() {
		return false
	}
	if opts == nil {
		return false
	}
	if v, ok := opts["prompt_cache_key"].(string); ok && strings.TrimSpace(v) != "" {
		return true
	}
	return agentPromptCacheKeyFromEliza(opts) != ""
}

// stableSystemPrefixHash returns a short hash of the first system message for
// L3 prefixHash metadata when operators send multi-turn chat without eliza hints.
func stableSystemPrefixHash(messages []api.Message) string {
	for _, m := range messages {
		if m.Role != "system" {
			continue
		}
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		sum := sha256.Sum256([]byte(content))
		return "sys:" + hex.EncodeToString(sum[:8])
	}
	return ""
}
