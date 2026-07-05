package server

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

func TestEnsureGeneratePromptCacheKey_elizaAndSystem(t *testing.T) {
	req := &api.GenerateRequest{
		System: "You are Gemma.",
		Options: map[string]any{
			"eliza": map[string]any{"conversationId": "thread-gen"},
		},
	}
	EnsureGeneratePromptCacheKey(req)
	if got := req.Options["prompt_cache_key"]; got != "conv:thread-gen" {
		t.Fatalf("prompt_cache_key = %v, want conv:thread-gen", got)
	}
	eliza := req.Options["eliza"].(map[string]any)
	if !strings.HasPrefix(eliza["prefixHash"].(string), "sys:") {
		t.Fatalf("prefixHash = %v", eliza["prefixHash"])
	}
}

func TestEnsureAgentPromptCacheKey_elizaConversation(t *testing.T) {
	req := &api.ChatRequest{
		Options: map[string]any{
			"eliza": map[string]any{
				"conversationId": "thread-abc",
			},
		},
	}
	EnsureAgentPromptCacheKey(req)
	if got := req.Options["prompt_cache_key"]; got != "conv:thread-abc" {
		t.Fatalf("prompt_cache_key = %v, want conv:thread-abc", got)
	}
}

func TestEnsureAgentPromptCacheKey_existingFlat(t *testing.T) {
	req := &api.ChatRequest{
		Options: map[string]any{"prompt_cache_key": "keep-me"},
	}
	EnsureAgentPromptCacheKey(req)
	if got := req.Options["prompt_cache_key"]; got != "keep-me" {
		t.Fatalf("prompt_cache_key = %v, want keep-me", got)
	}
	if _, ok := req.Options["eliza"]; ok {
		t.Fatal("generic key should not add eliza block")
	}
}

func TestEnsureAgentPromptCacheKey_backfillsEliza(t *testing.T) {
	req := &api.ChatRequest{
		Messages: []api.Message{{Role: "system", Content: "You are Gemma."}},
		Options:  map[string]any{"prompt_cache_key": "hermes:agent:main:cli:1"},
	}
	EnsureAgentPromptCacheKey(req)
	eliza, ok := req.Options["eliza"].(map[string]any)
	if !ok {
		t.Fatal("expected eliza block")
	}
	if eliza["promptCacheKey"] != "hermes:agent:main:cli:1" {
		t.Fatalf("promptCacheKey = %v", eliza["promptCacheKey"])
	}
	if eliza["conversationId"] != "agent:main:cli:1" {
		t.Fatalf("conversationId = %v", eliza["conversationId"])
	}
	if !strings.HasPrefix(eliza["prefixHash"].(string), "sys:") {
		t.Fatalf("prefixHash = %v", eliza["prefixHash"])
	}
}

func TestAgentCachePrefersRuntime(t *testing.T) {
	t.Setenv("ZEROLLAMA_AGENT_CACHE_RUNTIME", "1")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	if !agentCachePrefersRuntime(map[string]any{"prompt_cache_key": "agent-1"}) {
		t.Fatal("expected true when key set and env enabled")
	}
	if agentCachePrefersRuntime(nil) {
		t.Fatal("expected false without key")
	}
	t.Setenv("ZEROLLAMA_AGENT_CACHE_RUNTIME", "0")
	if agentCachePrefersRuntime(map[string]any{"prompt_cache_key": "agent-1"}) {
		t.Fatal("expected false when disabled")
	}
}

func TestAgentCacheRuntimeEnabledDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_AGENT_CACHE_RUNTIME", "")
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_EDGE", "0")
	if !envconfig.AgentCacheRuntimeEnabled() {
		t.Fatal("expected auto enable when runtime URL set")
	}
}

func TestStableSystemPrefixHash(t *testing.T) {
	h1 := stableSystemPrefixHash([]api.Message{{Role: "system", Content: "You are Gemma."}})
	h2 := stableSystemPrefixHash([]api.Message{{Role: "system", Content: "You are Gemma."}})
	if h1 == "" || h1 != h2 {
		t.Fatalf("hash = %q %q", h1, h2)
	}
	if !strings.HasPrefix(h1, "sys:") {
		t.Fatalf("want sys: prefix, got %q", h1)
	}
}
