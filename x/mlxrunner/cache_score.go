package mlxrunner

import (
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// beginScore prepares caches for System One scoring. Unlike generation begin,
// scoring may restore the complete prompt (no decode seed).
func (c *kvCache) beginScore(m base.Model, inputs []int32, items []scoreMediaItem) *cacheSession {
	c.ensureCaches(m)
	c.ensureRoot()

	keys := c.key(inputs)
	if len(items) > 0 {
		keys = c.keyScore(inputs, items)
	}
	matchPath, matched := findBestMatch(c.root, keys)
	originalMatched := matched

	if modelSlidingWindow(m) > 0 {
		matchPath, matched = capTrieMatchForRestore(matchPath, matched)
	}
	sameBranch := c.trySameBranchRestore(matchPath, matched)
	if !sameBranch {
		c.switchToPath(matchPath, matched)
	}

	prefix := c.minCacheOffset()
	remaining := inputs[prefix:]

	session := &cacheSession{
		cache:        c,
		inputs:       inputs,
		caches:       c.caches,
		remaining:    remaining,
		cachedPrefix: prefix,
		sameBranch:   sameBranch,
	}

	if prefix < matched {
		session.pendingSnapshots = append(session.pendingSnapshots, pendingSnapshot{offset: matched, user: false})
	}

	if prefix == 0 && originalMatched > 0 {
		// expected on first score row after a miss
	} else if prefix > 0 {
		_ = originalMatched
	}

	return session
}

func (c *kvCache) keyScore(tokens []int32, items []scoreMediaItem) []trieKey {
	eff := append([]int32(nil), tokens...)
	for _, item := range items {
		for i := item.pos; i < item.pos+item.length && i < len(eff); i++ {
			eff[i] = int32(item.fold)
		}
	}
	return c.key(eff)
}
