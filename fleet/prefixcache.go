package fleet

import (
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// PrefixCache maps (model, session_key) → node_id for fleet assign affinity.
// Complements per-node L3 prompt_cache_key pinning: route agent threads back to
// the node that recently served the same session.
type PrefixCache struct {
	mu      sync.RWMutex
	ttl     time.Duration
	entries map[string]prefixEntry
}

type prefixEntry struct {
	nodeID    string
	updatedAt time.Time
}

// NewPrefixCache creates a TTL-backed affinity index. ttl<=0 defaults to 30m.
func NewPrefixCache(ttl time.Duration) *PrefixCache {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	return &PrefixCache{
		ttl:     ttl,
		entries: make(map[string]prefixEntry),
	}
}

// PrefixCacheTTLFromEnv reads ZEROLLAMA_FLEET_PREFIX_CACHE_TTL (default 30m).
func PrefixCacheTTLFromEnv() time.Duration {
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_PREFIX_CACHE_TTL"))
	if s == "" {
		return 30 * time.Minute
	}
	if d, err := time.ParseDuration(s); err == nil && d > 0 {
		return d
	}
	if n, err := strconv.ParseInt(s, 10, 64); err == nil && n > 0 {
		return time.Duration(n) * time.Second
	}
	return 30 * time.Minute
}

func prefixCacheEnabled() bool {
	s := strings.TrimSpace(os.Getenv("ZEROLLAMA_FLEET_PREFIX_CACHE"))
	return s == "" || s == "1" || strings.EqualFold(s, "on") || strings.EqualFold(s, "true")
}

func newPrefixCacheFromEnv() *PrefixCache {
	if !prefixCacheEnabled() {
		return nil
	}
	return NewPrefixCache(PrefixCacheTTLFromEnv())
}

func prefixCacheKey(model, sessionKey string) string {
	return strings.ToLower(strings.TrimSpace(model)) + "\x00" + strings.TrimSpace(sessionKey)
}

// Remember records which node served a model+session pair.
func (c *PrefixCache) Remember(model, sessionKey, nodeID string) {
	if c == nil {
		return
	}
	sessionKey = strings.TrimSpace(sessionKey)
	nodeID = strings.TrimSpace(nodeID)
	if sessionKey == "" || nodeID == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[prefixCacheKey(model, sessionKey)] = prefixEntry{
		nodeID:    nodeID,
		updatedAt: time.Now().UTC(),
	}
}

// PreferredNode returns a recently assigned node for model+session when still valid.
func (c *PrefixCache) PreferredNode(model, sessionKey string) (nodeID string, ok bool) {
	if c == nil {
		return "", false
	}
	sessionKey = strings.TrimSpace(sessionKey)
	if sessionKey == "" {
		return "", false
	}
	now := time.Now().UTC()
	c.mu.RLock()
	e, found := c.entries[prefixCacheKey(model, sessionKey)]
	c.mu.RUnlock()
	if !found || e.nodeID == "" {
		return "", false
	}
	if c.ttl > 0 && now.Sub(e.updatedAt) > c.ttl {
		c.mu.Lock()
		delete(c.entries, prefixCacheKey(model, sessionKey))
		c.mu.Unlock()
		return "", false
	}
	return e.nodeID, true
}

// DropNode removes affinity entries pointing at an unavailable peer.
func (c *PrefixCache) DropNode(nodeID string) {
	if c == nil {
		return
	}
	nodeID = strings.ToLower(strings.TrimSpace(nodeID))
	if nodeID == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, e := range c.entries {
		if strings.EqualFold(e.nodeID, nodeID) {
			delete(c.entries, k)
		}
	}
}
