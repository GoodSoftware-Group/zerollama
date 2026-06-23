package mlxrunner

import (
	"hash/fnv"
	"sync"
	"time"
)

// tokenizeCache memoizes /v1/tokenize results per MLX runner client.
// Agent clients often send the same megaprompt every turn; chat truncation may
// tokenize the same rendered string more than once per request (message search +
// tail truncate). A small bounded LRU avoids redundant HTTP round-trips.
const (
	tokenizeCacheMaxEntries = 16
	tokenizeCacheMaxTokens  = 2_000_000 // ~8 MiB of []int backing storage
	tokenizeCacheTTL        = 15 * time.Minute
)

type tokenizeCache struct {
	mu          sync.Mutex
	entries     map[uint64]tokenizeCacheEntry
	totalTokens int
}

type tokenizeCacheEntry struct {
	length    int
	tokens    []int
	updatedAt time.Time
}

func tokenizeCacheKey(content string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(content))
	_, _ = h.Write([]byte{byte(len(content) >> 24), byte(len(content) >> 16), byte(len(content) >> 8), byte(len(content))})
	return h.Sum64()
}

func (c *tokenizeCache) lookup(content string) ([]int, bool) {
	if c == nil || content == "" {
		return nil, false
	}
	key := tokenizeCacheKey(content)
	now := time.Now().UTC()

	c.mu.Lock()
	defer c.mu.Unlock()

	e, ok := c.entries[key]
	if !ok || e.length != len(content) {
		return nil, false
	}
	if tokenizeCacheTTL > 0 && now.Sub(e.updatedAt) > tokenizeCacheTTL {
		c.removeLocked(key, e)
		return nil, false
	}
	e.updatedAt = now
	c.entries[key] = e
	out := make([]int, len(e.tokens))
	copy(out, e.tokens)
	return out, true
}

func (c *tokenizeCache) remember(content string, tokens []int) {
	if c == nil || content == "" || len(tokens) == 0 {
		return
	}
	stored := make([]int, len(tokens))
	copy(stored, tokens)
	key := tokenizeCacheKey(content)
	now := time.Now().UTC()

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.entries == nil {
		c.entries = make(map[uint64]tokenizeCacheEntry)
	}
	if old, ok := c.entries[key]; ok {
		c.totalTokens -= len(old.tokens)
	}
	for len(c.entries) >= tokenizeCacheMaxEntries || c.totalTokens+len(stored) > tokenizeCacheMaxTokens {
		if !c.evictOldestLocked(now) {
			break
		}
	}
	c.entries[key] = tokenizeCacheEntry{
		length:    len(content),
		tokens:    stored,
		updatedAt: now,
	}
	c.totalTokens += len(stored)
}

func (c *tokenizeCache) removeLocked(key uint64, e tokenizeCacheEntry) {
	delete(c.entries, key)
	c.totalTokens -= len(e.tokens)
}

func (c *tokenizeCache) evictOldestLocked(now time.Time) bool {
	var oldestKey uint64
	var oldest tokenizeCacheEntry
	found := false
	for k, e := range c.entries {
		if tokenizeCacheTTL > 0 && now.Sub(e.updatedAt) > tokenizeCacheTTL {
			c.removeLocked(k, e)
			return true
		}
		if !found || e.updatedAt.Before(oldest.updatedAt) {
			oldestKey = k
			oldest = e
			found = true
		}
	}
	if !found {
		return false
	}
	c.removeLocked(oldestKey, oldest)
	return true
}
