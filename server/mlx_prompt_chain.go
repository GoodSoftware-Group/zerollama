package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/modality"
)

// mlxPromptChainCache keeps stable render prefix + token IDs per agent thread.
// Gemma4 (and similar) append a generation stub at the end of every prefill
// (<|turn>model…). Turn 2 replaces that stub with the prior assistant reply,
// so full-render string prefix matching fails even when message history only
// grew append-only. We splice on the stable prefix (render minus gen stub).
const (
	mlxPromptChainMaxEntries = 16
	mlxPromptChainMaxTokens  = 2_000_000
	mlxPromptChainTTL        = 15 * time.Minute
)

// mlxGenPromptStubs are trailing generation-prompt suffixes stripped before
// stable-prefix comparison. Longest first.
var mlxGenPromptStubs = []string{
	"<|turn>model\n<|channel>thought\n<channel|>",
	"<|turn>model\n",
}

type mlxPromptChainEntry struct {
	stablePrefix   string
	stableTokens   []int
	msgCount       int
	msgFingerprint string
	updatedAt      time.Time
}

type mlxPromptChainCache struct {
	mu          sync.Mutex
	entries     map[string]mlxPromptChainEntry
	totalTokens int
}

var globalMLXPromptChain = mlxPromptChainCache{entries: make(map[string]mlxPromptChainEntry)}

type ctxKeyPromptCacheKey struct{}

func withPromptCacheKey(ctx context.Context, key string) context.Context {
	if key == "" {
		return ctx
	}
	return context.WithValue(ctx, ctxKeyPromptCacheKey{}, key)
}

func promptCacheKeyFromCtx(ctx context.Context) string {
	k, _ := ctx.Value(ctxKeyPromptCacheKey{}).(string)
	return k
}

func mlxRenderStablePrefix(rendered string) string {
	stable, _ := mlxSplitRenderGenStub(rendered)
	return stable
}

func mlxSplitRenderGenStub(rendered string) (stable, genStub string) {
	for _, stub := range mlxGenPromptStubs {
		if strings.HasSuffix(rendered, stub) {
			return rendered[:len(rendered)-len(stub)], stub
		}
	}
	return rendered, ""
}

func agentMessagesFingerprint(msgs []api.Message) string {
	if len(msgs) == 0 {
		return ""
	}
	h := sha256.New()
	for _, m := range msgs {
		_, _ = h.Write([]byte(m.Role))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(m.Content))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil)[:8])
}

func (c *mlxPromptChainCache) invalidate(key string) {
	// Drop stable-prefix token IDs for this thread. Called when messages_dropped > 0
	// because the cached prefix referred to messages no longer in the render.
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		return
	}
	if prev, ok := c.entries[key]; ok {
		c.totalTokens -= len(prev.stableTokens)
		delete(c.entries, key)
		slog.Debug("mlx prompt chain invalidated", "key", key)
	}
}

func (c *mlxPromptChainCache) lookup(key string) (mlxPromptChainEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		return mlxPromptChainEntry{}, false
	}
	entry, ok := c.entries[key]
	if !ok || entry.stablePrefix == "" || len(entry.stableTokens) == 0 {
		return mlxPromptChainEntry{}, false
	}
	if mlxPromptChainTTL > 0 && time.Since(entry.updatedAt) > mlxPromptChainTTL {
		delete(c.entries, key)
		c.totalTokens -= len(entry.stableTokens)
		return mlxPromptChainEntry{}, false
	}
	return entry, true
}

func mlxPromptChainMessagesExtend(cached mlxPromptChainEntry, msgs []api.Message) bool {
	if cached.msgCount <= 0 || len(msgs) < cached.msgCount {
		return false
	}
	if len(msgs) == cached.msgCount {
		// Equal count is not append-only: verify fingerprint so in-place edits
		// (compression, last-message rewrite) do not reuse stale stable tokens.
		return agentMessagesFingerprint(msgs) == cached.msgFingerprint
	}
	return agentMessagesFingerprint(msgs[:cached.msgCount]) == cached.msgFingerprint
}

// mlxPromptChainTokensForRender reuses cached stable-prefix token IDs when agent
// history grew append-only. Tokenizes only the new stable delta + generation stub.
func mlxPromptChainTokensForRender(ctx context.Context, key, rendered string, msgs []api.Message, tokenize tokenizeFunc) ([]int, bool) {
	if key == "" || rendered == "" || tokenize == nil {
		return nil, false
	}

	entry, ok := globalMLXPromptChain.lookup(key)
	if !ok {
		return nil, false
	}
	if !mlxPromptChainMessagesExtend(entry, msgs) {
		recordMLXChainMiss(key, "messages_prefix_mismatch", map[string]any{
			"cached_msgs": entry.msgCount,
			"want_msgs":   len(msgs),
		})
		return nil, false
	}

	stable, genStub := mlxSplitRenderGenStub(rendered)
	if len(stable) < len(entry.stablePrefix) || stable[:len(entry.stablePrefix)] != entry.stablePrefix {
		recordMLXChainMiss(key, "stable_prefix_mismatch", map[string]any{
			"cached_stable_len": len(entry.stablePrefix),
			"stable_len":        len(stable),
		})
		return nil, false
	}

	delta := stable[len(entry.stablePrefix):]
	suffix := delta + genStub
	if suffix == "" {
		out := make([]int, len(entry.stableTokens))
		copy(out, entry.stableTokens)
		if genStub != "" {
			stubIDs, err := tokenize(ctx, genStub)
			if err != nil {
				return nil, false
			}
			out = append(out, stubIDs...)
		}
		return out, true
	}

	suffixIDs, err := tokenize(ctx, suffix)
	if err != nil {
		recordMLXChainMiss(key, "suffix_tokenize", map[string]any{"error": err.Error()})
		return nil, false
	}

	out := make([]int, 0, len(entry.stableTokens)+len(suffixIDs))
	out = append(out, entry.stableTokens...)
	out = append(out, suffixIDs...)
	return out, true
}

func mlxPromptChainReconcile(key string, spliced, fresh []int) []int {
	if len(spliced) == 0 {
		return fresh
	}
	if len(fresh) == 0 || slices.Equal(spliced, fresh) {
		return spliced
	}
	lcp := 0
	for lcp < len(spliced) && lcp < len(fresh) && spliced[lcp] == fresh[lcp] {
		lcp++
	}
	if lcp < minInt(len(spliced), len(fresh)) && len(fresh) > 256 && lcp < 64 {
		slog.Warn("mlx prompt chain splice diverged from full tokenize early; using splice for cache alignment",
			"key", key, "spliced", len(spliced), "fresh", len(fresh), "lcp", lcp)
	} else {
		slog.Debug("mlx prompt chain splice differs from full tokenize at suffix",
			"key", key, "spliced", len(spliced), "fresh", len(fresh), "lcp", lcp)
	}
	return spliced
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func rememberMLXPromptChain(m *Model, opts map[string]any, rendered string, msgs []api.Message, tokenize tokenizeFunc) {
	if m == nil || !m.IsMLX() || tokenize == nil {
		return
	}
	key := modality.ExtractPromptCacheKey(opts)
	if key == "" || rendered == "" || len(msgs) == 0 {
		return
	}
	stable := mlxRenderStablePrefix(rendered)
	if stable == "" {
		return
	}
	stableIDs, err := tokenize(context.Background(), stable)
	if err != nil || len(stableIDs) == 0 {
		return
	}
	globalMLXPromptChain.remember(key, stable, stableIDs, msgs)
}

func (c *mlxPromptChainCache) remember(key, stablePrefix string, stableTokens []int, msgs []api.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = make(map[string]mlxPromptChainEntry)
	}

	if prev, ok := c.entries[key]; ok {
		c.totalTokens -= len(prev.stableTokens)
		delete(c.entries, key)
	}

	stored := make([]int, len(stableTokens))
	copy(stored, stableTokens)
	c.entries[key] = mlxPromptChainEntry{
		stablePrefix:   stablePrefix,
		stableTokens:   stored,
		msgCount:       len(msgs),
		msgFingerprint: agentMessagesFingerprint(msgs),
		updatedAt:      time.Now().UTC(),
	}
	c.totalTokens += len(stored)
	c.evictLocked()
}

func (c *mlxPromptChainCache) evictLocked() {
	for len(c.entries) > mlxPromptChainMaxEntries || c.totalTokens > mlxPromptChainMaxTokens {
		var oldestKey string
		var oldestTime time.Time
		first := true
		for k, e := range c.entries {
			if first || e.updatedAt.Before(oldestTime) {
				oldestKey = k
				oldestTime = e.updatedAt
				first = false
			}
		}
		if oldestKey == "" {
			break
		}
		if prev, ok := c.entries[oldestKey]; ok {
			c.totalTokens -= len(prev.stableTokens)
			delete(c.entries, oldestKey)
			slog.Debug("mlx prompt chain evicted", "key", oldestKey)
		}
	}
}

func recordMLXChainMiss(key, reason string, extra map[string]any) {
	slog.Debug("mlx prompt chain miss", "key", key, "reason", reason)
	fields := map[string]any{
		"prompt_cache_key": key,
		"reason":           reason,
	}
	for k, v := range extra {
		fields[k] = v
	}
	RecordAgentStatsEvent("prompt_chain_miss", fields)
}
