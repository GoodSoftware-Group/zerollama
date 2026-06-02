package server

import (
	"context"
	"errors"
	"sync"

	"github.com/ollama/ollama/internal/runtimeclient"
)

// ErrRuntimeTokenizeUnavailable means Python /internal/tokenize could not be used
// (runtime down, libllama missing, etc.). Callers may fall back to heuristic truncation.
// Why a distinct error: Phase 12 tools need token-accurate truncation when possible, but
// operators should still get a rendered prompt if the sidecar is down — not a hard 500
// unless tokenize was attempted and returned a real HTTP/model error.
var ErrRuntimeTokenizeUnavailable = errors.New("runtime tokenize unavailable")

// memoizeTokenize caches tokenize results per rendered prompt string for one render pass.
// Why: renderChatPromptTokenized binary-searches message prefixes; each step re-tokenizes
// the full rendered prompt. Without cache, Phase 14 would issue one loopback HTTP call
// per search step to Python /internal/tokenize.
func memoizeTokenize(inner tokenizeFunc) tokenizeFunc {
	if inner == nil {
		return nil
	}
	var mu sync.Mutex
	cache := make(map[string][]int)
	return func(ctx context.Context, content string) ([]int, error) {
		mu.Lock()
		if tokens, ok := cache[content]; ok {
			mu.Unlock()
			return tokens, nil
		}
		mu.Unlock()
		tokens, err := inner(ctx, content)
		if err != nil {
			return nil, err
		}
		mu.Lock()
		cache[content] = tokens
		mu.Unlock()
		return tokens, nil
	}
}

// tokenizeForRuntimeModel uses Python libllama vocab when no ggml runner is loaded.
// Why runtimeProxyConfigured (not envconfig.RuntimeURL alone): embedded runtime sets
// loopback base URL via runtimeworker.BaseURL() while ZEROLLAMA_RUNTIME_URL stays unset.
// Checking only the env var skipped tokenize under embed and left truncate_mode=heuristic.
func (s *Server) tokenizeForRuntimeModel(m *Model) tokenizeFunc {
	if s == nil || m == nil {
		return nil
	}
	if !runtimeProxyConfigured() {
		return nil
	}
	gguf := m.ModelPath
	if gguf == "" {
		var ok bool
		gguf, ok = runtimeGGUFPath(m.Name)
		if !ok {
			return nil
		}
	}
	ggufPath := gguf
	return func(ctx context.Context, content string) ([]int, error) {
		tokens, ok, err := runtimeclient.Tokenize(ctx, ggufPath, content, true)
		if err != nil {
			if errors.Is(err, runtimeclient.ErrUnavailable) {
				return nil, ErrRuntimeTokenizeUnavailable
			}
			return nil, err
		}
		if !ok {
			return nil, ErrRuntimeTokenizeUnavailable
		}
		return tokens, nil
	}
}
