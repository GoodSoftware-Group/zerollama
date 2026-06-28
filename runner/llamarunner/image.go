package llamarunner

import (
	"errors"
	"fmt"
	"hash/maphash"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llama"
)

const (
	// sessionEmbedMaxSessions bounds per-agent ViT overlay maps on a long-lived runner.
	// WHY: complements the small global LRU — a fleet can evict frames globally while
	// the same eliza thread (prompt_cache_key) still hits session embeds on turn 2+.
	sessionEmbedMaxSessions = 32
	sessionEmbedTTL         = 30 * time.Minute
)

// defaultImageCacheSize is the embed cache size used when the caller does not
// specify one. Matches the historical constant; operators running video agents
// should raise this via OLLAMA_IMAGE_EMBED_CACHE_SIZE.
const defaultImageCacheSize = 4

type sessionEmbedState struct {
	byHash            map[uint64][]llama.MtmdChunk
	precomputedByHash map[uint64][]visionChunk
	updatedAt         time.Time
}

type ImageContext struct {
	// mu is required to be held when generating embeddings or accessing the cache
	mu sync.Mutex

	mtmd *llama.MtmdContext

	// cache of images to embeddings
	images    []imageCache
	imageHash maphash.Hash

	// cache of precomputed embedding rows (SGLang precomputed_embedding path)
	precomputed []precomputedCache

	// session overlay keyed by prompt_cache_key (agent thread id)
	sessionEmbeds   map[string]sessionEmbedState
	sessionEmbedLRU []string
}

// NewImageContext creates the vision embed context. cacheSize controls how many
// distinct image embeddings are kept in the LRU; pass ≤0 to use defaultImageCacheSize.
func NewImageContext(llamaContext *llama.Context, modelPath string, cacheSize int) (*ImageContext, error) {
	arch, err := llama.GetModelArch(modelPath)
	if err != nil {
		return nil, fmt.Errorf("unable to determine vision architecture: %w (%s)", err, modelPath)
	}

	var c ImageContext
	if arch == "clip" {
		c.mtmd, err = llama.NewMtmdContext(llamaContext, modelPath)
	} else {
		return nil, fmt.Errorf("unknown vision model architecture: %s", arch)
	}

	if err != nil {
		return nil, err
	}

	if cacheSize <= 0 {
		cacheSize = defaultImageCacheSize
	}
	c.images = make([]imageCache, cacheSize)
	c.precomputed = make([]precomputedCache, cacheSize)

	return &c, nil
}

func (c *ImageContext) Free(modelPath string) {
	if c == nil {
		return
	}

	if c.mtmd != nil {
		c.mtmd.Free()
	}
}

func normalizeSessionKey(sessionKey string) string {
	return strings.TrimSpace(sessionKey)
}

// MultimodalTokenize returns ViT chunks for image bytes. sessionKey (prompt_cache_key)
// enables a per-agent overlay that survives global LRU eviction between agent turns.
//
// gridTHW is optional [1,H,W] from video_spans (per-frame after expansion). Passed through
// to mtmd when upstream accepts client patch grids; today used for debug compare only.
// WHY cache ignores gridTHW: embeds are keyed by raster bytes — same PNG → same ViT output
// regardless of whether the client attached a layout hint.
func (c *ImageContext) MultimodalTokenize(llamaContext *llama.Context, data []byte, sessionKey string, gridTHW []int, sessionOverlay bool) ([]llama.MtmdChunk, error) {
	if c == nil {
		return nil, nil
	}

	if len(data) <= 0 {
		return nil, errors.New("received zero length image")
	}

	hash := c.hashImage(data)
	sessionKey = normalizeSessionKey(sessionKey)

	c.mu.Lock()
	defer c.mu.Unlock()

	if sessionOverlay {
		if chunks, ok := c.findSessionEmbedLocked(sessionKey, hash); ok {
			return chunks, nil
		}
	}

	chunks, err := c.findImage(hash)
	if err != nil {
		if c.mtmd != nil {
			chunks, err = c.mtmd.MultimodalTokenize(llamaContext, data, gridTHW)
			if err != nil {
				return nil, err
			}
		} else {
			return nil, errors.New("received image but vision model not loaded")
		}

		c.addImage(hash, chunks)
	} else {
		slog.Info("vision embed global cache hit")
	}
	if sessionOverlay {
		c.storeSessionEmbedLocked(sessionKey, hash, chunks)
	}

	return chunks, nil
}

func (c *ImageContext) BatchSize(configuredBatchSize int) int {
	// If images are not supported, we don't need to allocate embedding batches
	if c == nil {
		return 0
	}

	return configuredBatchSize
}

func (c *ImageContext) EmbedSize(llamaContext *llama.Context) int {
	return llamaContext.Model().NEmbd()
}

type imageCache struct {
	key      uint64
	val      []llama.MtmdChunk
	lastUsed time.Time
}

func (c *ImageContext) hashImage(image []byte) uint64 {
	c.imageHash.Reset()
	_, _ = c.imageHash.Write(image)
	return c.imageHash.Sum64()
}

var errImageNotFound = errors.New("image not found in cache")

func (c *ImageContext) findImage(hash uint64) ([]llama.MtmdChunk, error) {
	for i := range c.images {
		if c.images[i].key == hash {
			slog.Debug("loading image embeddings from cache", "entry", i)
			c.images[i].lastUsed = time.Now()
			return c.images[i].val, nil
		}
	}

	return nil, errImageNotFound
}

func (c *ImageContext) addImage(hash uint64, embed []llama.MtmdChunk) {
	best := time.Now()
	var bestImage int

	for i := range c.images {
		if c.images[i].key == hash {
			bestImage = i
			break
		}

		if c.images[i].lastUsed.Compare(best) < 0 {
			best = c.images[i].lastUsed
			bestImage = i
		}
	}

	slog.Debug("storing image embeddings in cache", "entry", bestImage, "used", c.images[bestImage].lastUsed)
	c.images[bestImage].key = hash
	c.images[bestImage].val = embed
	c.images[bestImage].lastUsed = time.Now()
}

// growCacheForDistinctFrames expands the embed LRU when a multimodal turn has more
// rasters than initial slots. WHY: SGLang prefix-mm cache keeps all clip frames hot;
// operators should not have to restart runners or raise OLLAMA_IMAGE_EMBED_CACHE_SIZE
// before the first 32-frame video request.
func (c *ImageContext) growCacheForDistinctFrames(frameCount int) {
	if c == nil || frameCount <= len(c.images) {
		return
	}
	want := frameCount
	if max := envconfig.ImageEmbedCacheMax(); want > max {
		want = max
	}
	if want <= len(c.images) {
		return
	}
	next := make([]imageCache, want)
	copy(next, c.images)
	c.images = next
	if len(c.precomputed) < want {
		pc := make([]precomputedCache, want)
		copy(pc, c.precomputed)
		c.precomputed = pc
	}
	slog.Info("vision embed cache auto-grown for multimodal turn",
		"slots", want,
		"frames", frameCount,
	)
}

func (c *ImageContext) findSessionEmbedLocked(sessionKey string, hash uint64) ([]llama.MtmdChunk, bool) {
	if c == nil || sessionKey == "" {
		return nil, false
	}
	st, ok := c.sessionEmbeds[sessionKey]
	if !ok {
		return nil, false
	}
	if sessionEmbedTTL > 0 && time.Since(st.updatedAt) > sessionEmbedTTL {
		c.evictSessionEmbedLocked(sessionKey)
		return nil, false
	}
	chunks, ok := st.byHash[hash]
	if !ok {
		return nil, false
	}
	st.updatedAt = time.Now()
	c.sessionEmbeds[sessionKey] = st
	c.bumpSessionEmbedLRULocked(sessionKey)
	slog.Info("vision embed session cache hit", "session_key", sessionKey)
	return chunks, true
}

func (c *ImageContext) storeSessionEmbedLocked(sessionKey string, hash uint64, chunks []llama.MtmdChunk) {
	if c == nil || sessionKey == "" || len(chunks) == 0 {
		return
	}
	if c.sessionEmbeds == nil {
		c.sessionEmbeds = make(map[string]sessionEmbedState)
	}
	st, ok := c.sessionEmbeds[sessionKey]
	if !ok {
		if len(c.sessionEmbedLRU) >= sessionEmbedMaxSessions {
			c.evictSessionEmbedLocked(c.sessionEmbedLRU[0])
		}
		st = sessionEmbedState{
			byHash:            make(map[uint64][]llama.MtmdChunk),
			precomputedByHash: make(map[uint64][]visionChunk),
		}
		// Key is new: append to tail. No bump needed — it is already at the tail.
		c.sessionEmbedLRU = append(c.sessionEmbedLRU, sessionKey)
	} else {
		if st.byHash == nil {
			st.byHash = make(map[uint64][]llama.MtmdChunk)
		}
		if st.precomputedByHash == nil {
			st.precomputedByHash = make(map[uint64][]visionChunk)
		}
		// Existing session: move to tail so it is evicted last.
		c.bumpSessionEmbedLRULocked(sessionKey)
	}
	st.byHash[hash] = chunks
	st.updatedAt = time.Now()
	c.sessionEmbeds[sessionKey] = st
}

func (c *ImageContext) bumpSessionEmbedLRULocked(key string) {
	for i, k := range c.sessionEmbedLRU {
		if k == key {
			c.sessionEmbedLRU = append(append(c.sessionEmbedLRU[:i], c.sessionEmbedLRU[i+1:]...), key)
			return
		}
	}
}

func (c *ImageContext) evictSessionEmbedLocked(key string) {
	delete(c.sessionEmbeds, key)
	for i, k := range c.sessionEmbedLRU {
		if k == key {
			c.sessionEmbedLRU = append(c.sessionEmbedLRU[:i], c.sessionEmbedLRU[i+1:]...)
			return
		}
	}
}

func (c *ImageContext) findSessionPrecomputedLocked(sessionKey string, hash uint64) ([]visionChunk, bool) {
	if c == nil || sessionKey == "" {
		return nil, false
	}
	st, ok := c.sessionEmbeds[sessionKey]
	if !ok {
		return nil, false
	}
	if sessionEmbedTTL > 0 && time.Since(st.updatedAt) > sessionEmbedTTL {
		c.evictSessionEmbedLocked(sessionKey)
		return nil, false
	}
	vc, ok := st.precomputedByHash[hash]
	if !ok {
		return nil, false
	}
	st.updatedAt = time.Now()
	c.sessionEmbeds[sessionKey] = st
	c.bumpSessionEmbedLRULocked(sessionKey)
	slog.Info("precomputed_embedding session cache hit", "session_key", sessionKey)
	return cloneVisionChunks(vc), true
}

func (c *ImageContext) storeSessionPrecomputedLocked(sessionKey string, hash uint64, chunks []visionChunk) {
	if c == nil || sessionKey == "" || len(chunks) == 0 {
		return
	}
	if c.sessionEmbeds == nil {
		c.sessionEmbeds = make(map[string]sessionEmbedState)
	}
	st, ok := c.sessionEmbeds[sessionKey]
	if !ok {
		if len(c.sessionEmbedLRU) >= sessionEmbedMaxSessions {
			c.evictSessionEmbedLocked(c.sessionEmbedLRU[0])
		}
		st = sessionEmbedState{
			byHash:            make(map[uint64][]llama.MtmdChunk),
			precomputedByHash: make(map[uint64][]visionChunk),
		}
		c.sessionEmbedLRU = append(c.sessionEmbedLRU, sessionKey)
	} else {
		if st.precomputedByHash == nil {
			st.precomputedByHash = make(map[uint64][]visionChunk)
		}
		c.bumpSessionEmbedLRULocked(sessionKey)
	}
	st.precomputedByHash[hash] = cloneVisionChunks(chunks)
	st.updatedAt = time.Now()
	c.sessionEmbeds[sessionKey] = st
}
