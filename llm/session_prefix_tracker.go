package llm

import (
	"sync"
)

// sessionPrefixTracker remembers the last pretokenized prompt per model+session key
// so turn N+1 can estimate num_computed before llama-server IPC (vLLM #52041).
type sessionPrefixTracker struct {
	mu    sync.Mutex
	byKey map[string][]int
}

var llamaSessionPrefixTracker sessionPrefixTracker

func (t *sessionPrefixTracker) key(modelPath, promptCacheKey string) string {
	return modelPath + "\x00" + promptCacheKey
}

func tokensEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func isStrictPrefix(prefix, tokens []int) bool {
	if len(prefix) == 0 || len(prefix) >= len(tokens) {
		return false
	}
	for i := range prefix {
		if prefix[i] != tokens[i] {
			return false
		}
	}
	return true
}

func (t *sessionPrefixTracker) estimate(modelPath, promptCacheKey string, promptTokens []int, cacheReset bool) int {
	if cacheReset || promptCacheKey == "" || len(promptTokens) == 0 {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.byKey == nil {
		return 0
	}
	prev, ok := t.byKey[t.key(modelPath, promptCacheKey)]
	if !ok || len(prev) <= 1 {
		return 0
	}
	if !isStrictPrefix(prev, promptTokens) && !tokensEqual(prev, promptTokens) {
		return 0
	}
	return len(prev) - 1
}

func (t *sessionPrefixTracker) record(modelPath, promptCacheKey string, promptTokens []int, cacheReset bool) {
	if promptCacheKey == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.byKey == nil {
		t.byKey = make(map[string][]int)
	}
	sk := t.key(modelPath, promptCacheKey)
	if cacheReset {
		delete(t.byKey, sk)
		return
	}
	if len(promptTokens) == 0 {
		return
	}
	t.byKey[sk] = append([]int(nil), promptTokens...)
}
