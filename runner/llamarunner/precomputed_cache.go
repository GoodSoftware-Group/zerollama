package llamarunner

import (
	"errors"
	"log/slog"
	"time"
)

type precomputedCache struct {
	key      uint64
	val      []visionChunk
	bytes    int64
	lastUsed time.Time
}

var errPrecomputedNotFound = errors.New("precomputed embedding not found in cache")

// GetPrecomputedChunks materializes vision chunks from SGLang precomputed rows with
// global LRU + optional session overlay (mirrors PNG MultimodalTokenize caching).
func (c *ImageContext) GetPrecomputedChunks(rows [][]float32, sessionKey string, sessionOverlay bool) ([]visionChunk, error) {
	if len(rows) == 0 {
		return nil, errors.New("precomputed feature is empty")
	}
	if c == nil {
		return visionChunksFromPrecomputed(rows), nil
	}

	hash := hashPrecomputedRows(rows)
	sessionKey = normalizeSessionKey(sessionKey)

	c.mu.Lock()
	defer c.mu.Unlock()

	if sessionOverlay {
		if vc, ok := c.findSessionPrecomputedLocked(sessionKey, hash); ok {
			return vc, nil
		}
	}
	if vc, err := c.findPrecomputedLocked(hash); err == nil {
		if sessionOverlay && sessionKey != "" {
			c.storeSessionPrecomputedLocked(sessionKey, hash, vc)
		}
		if c.byteBudget > 0 {
			slog.Info("precomputed_embedding radix cache hit")
		}
		slog.Info("precomputed_embedding global cache hit")
		return cloneVisionChunks(vc), nil
	}

	vc := visionChunksFromPrecomputed(rows)
	c.addPrecomputedLocked(hash, vc)
	if sessionOverlay && sessionKey != "" {
		c.storeSessionPrecomputedLocked(sessionKey, hash, vc)
	}
	slog.Info("precomputed_embedding runner inject",
		"rows", len(rows),
		"hidden", len(rows[0]),
	)
	return cloneVisionChunks(vc), nil
}

func (c *ImageContext) findPrecomputedLocked(hash uint64) ([]visionChunk, error) {
	for i := range c.precomputed {
		if c.precomputed[i].key == hash {
			c.precomputed[i].lastUsed = time.Now()
			return c.precomputed[i].val, nil
		}
	}
	return nil, errPrecomputedNotFound
}

func (c *ImageContext) addPrecomputedLocked(hash uint64, vc []visionChunk) {
	stored := cloneVisionChunks(vc)
	need := visionChunksBytes(stored)

	for i := range c.precomputed {
		if c.precomputed[i].key == hash {
			c.totalBytes -= c.precomputed[i].bytes
			if c.totalBytes < 0 {
				c.totalBytes = 0
			}
			c.precomputed[i].val = stored
			c.precomputed[i].bytes = need
			c.precomputed[i].lastUsed = time.Now()
			c.totalBytes += need
			c.evictUntilBudgetLocked(0)
			return
		}
	}

	c.evictUntilBudgetLocked(need)

	bestSlot := -1
	for i := range c.precomputed {
		if c.precomputed[i].key == 0 && len(c.precomputed[i].val) == 0 {
			bestSlot = i
			break
		}
	}

	if bestSlot < 0 && c.canGrowRadixPrecomputedLocked(need) {
		c.precomputed = append(c.precomputed, precomputedCache{})
		bestSlot = len(c.precomputed) - 1
		slog.Debug("vision embed radix pool grown",
			"pool", "precomputed",
			"slots", len(c.precomputed),
			"bytes", need,
		)
	}

	if bestSlot < 0 {
		best := time.Now()
		bestSlot = 0
		for i := range c.precomputed {
			if c.precomputed[i].lastUsed.Compare(best) < 0 {
				best = c.precomputed[i].lastUsed
				bestSlot = i
			}
		}
		if c.precomputed[bestSlot].key != 0 || len(c.precomputed[bestSlot].val) != 0 {
			c.clearPrecomputedEntryLocked(bestSlot)
		}
	}

	c.precomputed[bestSlot].key = hash
	c.precomputed[bestSlot].val = stored
	c.precomputed[bestSlot].bytes = need
	c.precomputed[bestSlot].lastUsed = time.Now()
	c.totalBytes += need
}

func (c *ImageContext) canGrowRadixPrecomputedLocked(need int64) bool {
	if c.byteBudget <= 0 || len(c.precomputed) >= vitRadixHardSlotCap {
		return false
	}
	return c.totalBytes+need <= c.byteBudget
}
