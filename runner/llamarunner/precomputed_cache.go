package llamarunner

import (
	"errors"
	"log/slog"
	"time"
)

type precomputedCache struct {
	key      uint64
	val      []visionChunk
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
	best := time.Now()
	var bestSlot int

	for i := range c.precomputed {
		if c.precomputed[i].key == hash {
			bestSlot = i
			break
		}
		if c.precomputed[i].lastUsed.Compare(best) < 0 {
			best = c.precomputed[i].lastUsed
			bestSlot = i
		}
	}

	c.precomputed[bestSlot].key = hash
	c.precomputed[bestSlot].val = cloneVisionChunks(vc)
	c.precomputed[bestSlot].lastUsed = time.Now()
}
