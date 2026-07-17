package llamarunner

import (
	"encoding/binary"
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
	// vitRadixHardSlotCap bounds grow-under-budget entry metadata (byte budget caps floats).
	vitRadixHardSlotCap = 4096
)

// defaultImageCacheSize is the embed cache size used when the caller does not
// specify one. Matches the historical constant; operators running video agents
// should raise this via OLLAMA_IMAGE_EMBED_CACHE_SIZE.
const defaultImageCacheSize = 4

type sessionEmbedState struct {
	byHash            map[uint64][]llama.MtmdChunk
	hashLRU           []uint64
	precomputedByHash map[uint64][]visionChunk
	precomputedLRU    []uint64
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

	// Byte budget across images + precomputed (EffectiveImageEmbedCacheBytes / ViT radix).
	byteBudget int64
	totalBytes int64

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
	c.byteBudget = envconfig.EffectiveImageEmbedCacheBytes()
	if c.byteBudget > 0 {
		slog.Info("vision embed radix pool enabled",
			"byte_budget", c.byteBudget,
			"slots", cacheSize,
		)
	}

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
// gridTHW is optional [1,H,W] from video_spans (per-frame after expansion). When set,
// mtmd dyn_size honors it via mtmd_bitmap_set_grid_hint (M-RoPE / Qwen-VL).
// Cache keys include grid when present — same PNG + different grid → different embeds.
func (c *ImageContext) MultimodalTokenize(llamaContext *llama.Context, data []byte, sessionKey string, gridTHW []int, sessionOverlay bool) ([]llama.MtmdChunk, error) {
	if c == nil {
		return nil, nil
	}

	if len(data) <= 0 {
		return nil, errors.New("received zero length image")
	}

	hash := c.hashImage(data, gridTHW)
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
	} else if c.byteBudget > 0 {
		slog.Info("vision embed radix cache hit")
		slog.Info("vision embed global cache hit")
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
	bytes    int64
	lastUsed time.Time
}

func mtmdChunksBytes(chunks []llama.MtmdChunk) int64 {
	var n int64
	for _, ch := range chunks {
		n += int64(len(ch.Embed)) * 4
	}
	return n
}

func visionChunksBytes(chunks []visionChunk) int64 {
	var n int64
	for _, ch := range chunks {
		n += int64(len(ch.embed)) * 4
	}
	return n
}

func (c *ImageContext) hashImage(image []byte, gridTHW []int) uint64 {
	c.imageHash.Reset()
	_, _ = c.imageHash.Write(image)
	if len(gridTHW) == 3 {
		var buf [24]byte
		binary.LittleEndian.PutUint64(buf[0:8], uint64(gridTHW[0]))
		binary.LittleEndian.PutUint64(buf[8:16], uint64(gridTHW[1]))
		binary.LittleEndian.PutUint64(buf[16:24], uint64(gridTHW[2]))
		_, _ = c.imageHash.Write(buf[:])
	}
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

func (c *ImageContext) clearImageEntryLocked(i int) {
	c.totalBytes -= c.images[i].bytes
	if c.totalBytes < 0 {
		c.totalBytes = 0
	}
	c.images[i] = imageCache{}
}

func (c *ImageContext) clearPrecomputedEntryLocked(i int) {
	c.totalBytes -= c.precomputed[i].bytes
	if c.totalBytes < 0 {
		c.totalBytes = 0
	}
	c.precomputed[i] = precomputedCache{}
}

// oldestFilledGlobalLocked returns which pool ("image"|"precomputed") and index of the
// least-recently-used filled slot across both LRUs. Empty → ("", -1).
func (c *ImageContext) oldestFilledGlobalLocked() (pool string, idx int) {
	best := -1
	bestPool := ""
	var bestTime time.Time
	for i := range c.images {
		if c.images[i].key == 0 && len(c.images[i].val) == 0 {
			continue
		}
		if best < 0 || c.images[i].lastUsed.Compare(bestTime) < 0 {
			best = i
			bestTime = c.images[i].lastUsed
			bestPool = "image"
		}
	}
	for i := range c.precomputed {
		if c.precomputed[i].key == 0 && len(c.precomputed[i].val) == 0 {
			continue
		}
		if best < 0 || c.precomputed[i].lastUsed.Compare(bestTime) < 0 {
			best = i
			bestTime = c.precomputed[i].lastUsed
			bestPool = "precomputed"
		}
	}
	return bestPool, best
}

func (c *ImageContext) evictUntilBudgetLocked(need int64) {
	if c.byteBudget <= 0 {
		return
	}
	for c.totalBytes+need > c.byteBudget {
		pool, i := c.oldestFilledGlobalLocked()
		if i < 0 {
			return
		}
		if pool == "image" {
			c.clearImageEntryLocked(i)
		} else {
			c.clearPrecomputedEntryLocked(i)
		}
	}
}

func (c *ImageContext) addImage(hash uint64, embed []llama.MtmdChunk) {
	need := mtmdChunksBytes(embed)

	for i := range c.images {
		if c.images[i].key == hash {
			c.totalBytes -= c.images[i].bytes
			if c.totalBytes < 0 {
				c.totalBytes = 0
			}
			c.images[i].val = embed
			c.images[i].bytes = need
			c.images[i].lastUsed = time.Now()
			c.totalBytes += need
			c.evictUntilBudgetLocked(0)
			return
		}
	}

	c.evictUntilBudgetLocked(need)

	bestImage := -1
	for i := range c.images {
		if c.images[i].key == 0 && len(c.images[i].val) == 0 {
			bestImage = i
			break
		}
	}

	if bestImage < 0 && c.canGrowRadixImageLocked(need) {
		c.images = append(c.images, imageCache{})
		bestImage = len(c.images) - 1
		slog.Debug("vision embed radix pool grown",
			"pool", "image",
			"slots", len(c.images),
			"bytes", need,
		)
	}

	if bestImage < 0 {
		best := time.Now()
		bestImage = 0
		for i := range c.images {
			if c.images[i].lastUsed.Compare(best) < 0 {
				best = c.images[i].lastUsed
				bestImage = i
			}
		}
		if c.images[bestImage].key != 0 || len(c.images[bestImage].val) != 0 {
			c.clearImageEntryLocked(bestImage)
		}
	}

	slog.Debug("storing image embeddings in cache",
		"entry", bestImage,
		"bytes", need,
		"total_bytes", c.totalBytes+need,
	)
	c.images[bestImage].key = hash
	c.images[bestImage].val = embed
	c.images[bestImage].bytes = need
	c.images[bestImage].lastUsed = time.Now()
	c.totalBytes += need
}

func (c *ImageContext) canGrowRadixImageLocked(need int64) bool {
	if c.byteBudget <= 0 || len(c.images) >= vitRadixHardSlotCap {
		return false
	}
	return c.totalBytes+need <= c.byteBudget
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
	bumpHashLRU(&st.hashLRU, hash)
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
		c.sessionEmbedLRU = append(c.sessionEmbedLRU, sessionKey)
	} else {
		if st.byHash == nil {
			st.byHash = make(map[uint64][]llama.MtmdChunk)
		}
		if st.precomputedByHash == nil {
			st.precomputedByHash = make(map[uint64][]visionChunk)
		}
		c.bumpSessionEmbedLRULocked(sessionKey)
	}
	_, exists := st.byHash[hash]
	if !exists {
		maxHashes := envconfig.ImageEmbedCacheMax()
		for len(st.byHash) >= maxHashes && len(st.hashLRU) > 0 {
			victim := st.hashLRU[0]
			st.hashLRU = st.hashLRU[1:]
			delete(st.byHash, victim)
		}
	}
	st.byHash[hash] = chunks
	bumpHashLRU(&st.hashLRU, hash)
	st.updatedAt = time.Now()
	c.sessionEmbeds[sessionKey] = st
}

func bumpHashLRU(lru *[]uint64, hash uint64) {
	for i, h := range *lru {
		if h == hash {
			*lru = append(append((*lru)[:i], (*lru)[i+1:]...), hash)
			return
		}
	}
	*lru = append(*lru, hash)
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
	bumpHashLRU(&st.precomputedLRU, hash)
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
	_, exists := st.precomputedByHash[hash]
	if !exists {
		maxHashes := envconfig.ImageEmbedCacheMax()
		for len(st.precomputedByHash) >= maxHashes && len(st.precomputedLRU) > 0 {
			victim := st.precomputedLRU[0]
			st.precomputedLRU = st.precomputedLRU[1:]
			delete(st.precomputedByHash, victim)
		}
	}
	// Share with global slot — clone only on return to the request path.
	st.precomputedByHash[hash] = chunks
	bumpHashLRU(&st.precomputedLRU, hash)
	st.updatedAt = time.Now()
	c.sessionEmbeds[sessionKey] = st
}
