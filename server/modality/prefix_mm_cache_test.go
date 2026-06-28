package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestOptionsEnablePrefixMMCache(t *testing.T) {
	if !OptionsEnablePrefixMMCache(map[string]any{"enable_prefix_mm_cache": true}) {
		t.Fatal("expected true")
	}
	if !OptionsEnablePrefixMMCache(map[string]any{"enable_prefix_mm_cache": "true"}) {
		t.Fatal("expected string true")
	}
	if OptionsEnablePrefixMMCache(map[string]any{"enable_prefix_mm_cache": false}) {
		t.Fatal("expected false")
	}
}

func TestSessionViTOverlayEnabled(t *testing.T) {
	if SessionViTOverlayEnabled(map[string]any{"prompt_cache_key": "a"}) != true {
		t.Fatal("expected overlay on with session key")
	}
	if SessionViTOverlayEnabled(map[string]any{
		"prompt_cache_key":         "a",
		"enable_prefix_mm_cache": false,
	}) {
		t.Fatal("expected overlay off when enable_prefix_mm_cache=false")
	}
	if SessionViTOverlayEnabled(map[string]any{"enable_prefix_mm_cache": true}) {
		t.Fatal("expected overlay off without session key")
	}
}

func TestWarnPrefixMMCacheWithoutSessionKey_noPanic(t *testing.T) {
	WarnPrefixMMCacheWithoutSessionKey(&api.ChatRequest{
		Options: map[string]any{"enable_prefix_mm_cache": true},
	})
	WarnPrefixMMCacheWithoutSessionKey(&api.ChatRequest{
		Options: map[string]any{
			"enable_prefix_mm_cache": true,
			"prompt_cache_key":       "agent-1",
		},
	})
}
