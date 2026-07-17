package ollamarunner

import (
	"encoding/binary"
	"errors"
	"hash/maphash"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

const (
	sessionEmbedMaxSessions = 32
	sessionEmbedTTL         = 30 * time.Minute
	defaultImageCacheSize   = 4
	// vitRadixHardSlotCap bounds grow-under-budget so a pathological flood cannot
	// allocate unbounded entry metadata (byte budget still caps float payload).
	vitRadixHardSlotCap = 4096
)

var errVisionEmbedNotFound = errors.New("vision embed not found in cache")

type sessionEmbedState struct {
	byHash    map[uint64]cachedMultimodal
	hashLRU   []uint64
	updatedAt time.Time
}

type imageEmbedCache struct {
	key      uint64
	val      cachedMultimodal
	bytes    int64
	lastUsed time.Time
}

type cachedMultimodalPart struct {
	data   any
	dtype  ml.DType
	shape  []int
	floats []float32
}

type cachedMultimodal struct {
	parts []cachedMultimodalPart
}

func cachedMultimodalBytes(embed cachedMultimodal) int64 {
	var n int64
	for _, p := range embed.parts {
		n += int64(len(p.floats)) * 4
	}
	return n
}

// VisionEmbedCache stores materialized vision encoder outputs (float32 + metadata)
// so agent threads can skip re-encoding the same clip frames between turns.
//
// Capacity: slot LRU (OLLAMA_IMAGE_EMBED_CACHE_SIZE/MAX) plus byte budget
// (EffectiveImageEmbedCacheBytes — SGLang MultiModalStaticCache / #28441).
// When a byte budget is set (radix on by default), the content-addressed pool
// grows beyond MAX under budget so embeds survive across prompt_cache_key values.
type VisionEmbedCache struct {
	mu sync.Mutex

	entries          []imageEmbedCache
	totalBytes       int64
	byteBudget       int64
	imageHash        maphash.Hash
	sessionEmbeds    map[string]sessionEmbedState
	sessionEmbedLRU  []string
	sessionMaxHashes int
}

func NewVisionEmbedCache(cacheSize int) *VisionEmbedCache {
	if cacheSize <= 0 {
		cacheSize = defaultImageCacheSize
	}
	budget := envconfig.EffectiveImageEmbedCacheBytes()
	c := &VisionEmbedCache{
		entries:          make([]imageEmbedCache, cacheSize),
		byteBudget:       budget,
		sessionMaxHashes: envconfig.ImageEmbedCacheMax(),
	}
	if budget > 0 {
		slog.Info("vision embed radix pool enabled",
			"byte_budget", budget,
			"slots", cacheSize,
			"engine", "ollama",
		)
	}
	return c
}

func (c *VisionEmbedCache) radixPool() bool {
	return c != nil && c.byteBudget > 0
}

func logVisionEmbedPoolHit(engine string) {
	// Radix = cross-request content pool; keep "global" wording for older greps via dual log.
	slog.Info("vision embed radix cache hit", "engine", engine)
	slog.Info("vision embed global cache hit", "engine", engine)
}

func normalizeSessionKey(sessionKey string) string {
	return strings.TrimSpace(sessionKey)
}

func (c *VisionEmbedCache) hashImage(image []byte, gridTHW []int) uint64 {
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

func encodeMultimodalOptionalGrid(mp model.MultimodalProcessor, ctx ml.Context, data []byte, gridTHW []int) ([]input.Multimodal, error) {
	if len(gridTHW) == 3 {
		if g, ok := mp.(model.GridHintMultimodalProcessor); ok {
			return g.EncodeMultimodalWithGrid(ctx, data, gridTHW)
		}
	}
	return mp.EncodeMultimodal(ctx, data)
}

// growCacheForDistinctFrames expands the embed LRU when a multimodal turn has more
// rasters than initial slots (mirrors llamarunner ImageContext).
func (c *VisionEmbedCache) growCacheForDistinctFrames(frameCount int) {
	if c == nil || frameCount <= len(c.entries) {
		return
	}
	want := frameCount
	if max := envconfig.ImageEmbedCacheMax(); want > max {
		want = max
	}
	if want <= len(c.entries) {
		return
	}
	next := make([]imageEmbedCache, want)
	copy(next, c.entries)
	c.entries = next
	slog.Info("vision embed cache auto-grown for multimodal turn",
		"slots", want,
		"frames", frameCount,
		"engine", "ollama",
	)
}

func (c *VisionEmbedCache) GetOrEncode(
	mp model.MultimodalProcessor,
	backend ml.Backend,
	ctx ml.Context,
	data []byte,
	gridTHW []int,
	sessionKey string,
	sessionOverlay bool,
) ([]input.Multimodal, error) {
	if c == nil {
		return encodeMultimodalOptionalGrid(mp, ctx, data, gridTHW)
	}
	if len(data) == 0 {
		return nil, errors.New("received zero length image")
	}

	hash := c.hashImage(data, gridTHW)
	sessionKey = normalizeSessionKey(sessionKey)

	c.mu.Lock()
	if sessionOverlay && sessionKey != "" {
		if cached, ok := c.findSessionEmbedLocked(sessionKey, hash); ok {
			c.mu.Unlock()
			return restoreMultimodal(ctx, cached), nil
		}
	}
	if cached, err := c.findGlobalLocked(hash); err == nil {
		if sessionOverlay && sessionKey != "" {
			c.storeSessionEmbedLocked(sessionKey, hash, cached)
		}
		c.mu.Unlock()
		if c.radixPool() {
			logVisionEmbedPoolHit("ollama")
		} else {
			slog.Info("vision embed global cache hit", "engine", "ollama")
		}
		return restoreMultimodal(ctx, cached), nil
	}
	c.mu.Unlock()

	encodeCtx := backend.NewContext()
	mm, err := encodeMultimodalOptionalGrid(mp, encodeCtx, data, gridTHW)
	if err != nil {
		encodeCtx.Close()
		return nil, err
	}
	cached, err := materializeMultimodal(backend, mm)
	encodeCtx.Close()
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	c.addGlobalLocked(hash, cached)
	if sessionOverlay && sessionKey != "" {
		c.storeSessionEmbedLocked(sessionKey, hash, cached)
	}
	c.mu.Unlock()

	return restoreMultimodal(ctx, cached), nil
}

func materializeMultimodal(backend ml.Backend, mm []input.Multimodal) (cachedMultimodal, error) {
	var tensors []ml.Tensor
	for _, m := range mm {
		if m.Tensor != nil {
			tensors = append(tensors, m.Tensor)
		}
	}
	if len(tensors) == 0 {
		out := cachedMultimodal{parts: make([]cachedMultimodalPart, len(mm))}
		for i, m := range mm {
			out.parts[i].data = m.Data
		}
		return out, nil
	}

	computeCtx := backend.NewContext()
	defer computeCtx.Close()

	computeCtx.Forward(tensors...)
	computeCtx.SetBatchSize(512)
	computeCtx.Compute(tensors...)

	out := cachedMultimodal{parts: make([]cachedMultimodalPart, len(mm))}
	for i, m := range mm {
		out.parts[i].data = m.Data
		if m.Tensor == nil {
			continue
		}
		out.parts[i].dtype = m.Tensor.DType()
		out.parts[i].shape = append([]int(nil), m.Tensor.Shape()...)
		out.parts[i].floats = append([]float32(nil), m.Tensor.Floats()...)
	}
	return out, nil
}

func restoreMultimodal(ctx ml.Context, cached cachedMultimodal) []input.Multimodal {
	out := make([]input.Multimodal, len(cached.parts))
	for i, p := range cached.parts {
		out[i].Data = p.data
		if len(p.floats) == 0 {
			continue
		}
		shape := append([]int(nil), p.shape...)
		out[i].Tensor = ctx.Input().FromFloats(append([]float32(nil), p.floats...), shape...)
	}
	return out
}

func (c *VisionEmbedCache) findGlobalLocked(hash uint64) (cachedMultimodal, error) {
	for i := range c.entries {
		if c.entries[i].key == hash {
			slog.Debug("loading image embeddings from cache",
				"entry", i,
				"engine", "ollama",
			)
			c.entries[i].lastUsed = time.Now()
			return c.entries[i].val, nil
		}
	}
	return cachedMultimodal{}, errVisionEmbedNotFound
}

func (c *VisionEmbedCache) clearEntryLocked(i int) {
	c.totalBytes -= c.entries[i].bytes
	if c.totalBytes < 0 {
		c.totalBytes = 0
	}
	c.entries[i] = imageEmbedCache{}
}

func (c *VisionEmbedCache) oldestFilledIndexLocked() int {
	best := -1
	var bestTime time.Time
	for i := range c.entries {
		if c.entries[i].key == 0 && len(c.entries[i].val.parts) == 0 {
			continue
		}
		if best < 0 || c.entries[i].lastUsed.Compare(bestTime) < 0 {
			best = i
			bestTime = c.entries[i].lastUsed
		}
	}
	return best
}

func (c *VisionEmbedCache) addGlobalLocked(hash uint64, embed cachedMultimodal) {
	need := cachedMultimodalBytes(embed)

	for i := range c.entries {
		if c.entries[i].key == hash {
			c.totalBytes -= c.entries[i].bytes
			if c.totalBytes < 0 {
				c.totalBytes = 0
			}
			c.entries[i].val = embed
			c.entries[i].bytes = need
			c.entries[i].lastUsed = time.Now()
			c.totalBytes += need
			c.evictUntilBudgetLocked(0)
			return
		}
	}

	c.evictUntilBudgetLocked(need)

	bestEntry := -1
	for i := range c.entries {
		if c.entries[i].key == 0 && len(c.entries[i].val.parts) == 0 {
			bestEntry = i
			break
		}
	}

	if bestEntry < 0 && c.canGrowRadixLocked(need) {
		c.entries = append(c.entries, imageEmbedCache{})
		bestEntry = len(c.entries) - 1
		slog.Debug("vision embed radix pool grown",
			"slots", len(c.entries),
			"bytes", need,
			"engine", "ollama",
		)
	}

	if bestEntry < 0 {
		best := time.Now()
		bestEntry = 0
		for i := range c.entries {
			if c.entries[i].lastUsed.Compare(best) < 0 {
				best = c.entries[i].lastUsed
				bestEntry = i
			}
		}
		if c.entries[bestEntry].key != 0 || len(c.entries[bestEntry].val.parts) != 0 {
			c.clearEntryLocked(bestEntry)
		}
	}

	slog.Debug("storing image embeddings in cache",
		"entry", bestEntry,
		"bytes", need,
		"total_bytes", c.totalBytes+need,
		"engine", "ollama",
	)
	c.entries[bestEntry].key = hash
	c.entries[bestEntry].val = embed
	c.entries[bestEntry].bytes = need
	c.entries[bestEntry].lastUsed = time.Now()
	c.totalBytes += need
}

func (c *VisionEmbedCache) canGrowRadixLocked(need int64) bool {
	if c.byteBudget <= 0 || len(c.entries) >= vitRadixHardSlotCap {
		return false
	}
	return c.totalBytes+need <= c.byteBudget
}

// evictUntilBudgetLocked frees oldest filled slots until need fits under byteBudget.
func (c *VisionEmbedCache) evictUntilBudgetLocked(need int64) {
	if c.byteBudget <= 0 {
		return
	}
	for c.totalBytes+need > c.byteBudget {
		i := c.oldestFilledIndexLocked()
		if i < 0 {
			return
		}
		c.clearEntryLocked(i)
	}
}

func (c *VisionEmbedCache) findSessionEmbedLocked(sessionKey string, hash uint64) (cachedMultimodal, bool) {
	if sessionKey == "" {
		return cachedMultimodal{}, false
	}
	st, ok := c.sessionEmbeds[sessionKey]
	if !ok {
		return cachedMultimodal{}, false
	}
	if sessionEmbedTTL > 0 && time.Since(st.updatedAt) > sessionEmbedTTL {
		c.evictSessionEmbedLocked(sessionKey)
		return cachedMultimodal{}, false
	}
	cached, ok := st.byHash[hash]
	if !ok {
		return cachedMultimodal{}, false
	}
	st.updatedAt = time.Now()
	bumpHashLRU(&st.hashLRU, hash)
	c.sessionEmbeds[sessionKey] = st
	c.bumpSessionEmbedLRULocked(sessionKey)
	slog.Info("vision embed session cache hit",
		"session_key", sessionKey,
		"engine", "ollama",
	)
	return cached, true
}

func (c *VisionEmbedCache) storeSessionEmbedLocked(sessionKey string, hash uint64, embed cachedMultimodal) {
	if sessionKey == "" || len(embed.parts) == 0 {
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
		st = sessionEmbedState{byHash: make(map[uint64]cachedMultimodal)}
		c.sessionEmbedLRU = append(c.sessionEmbedLRU, sessionKey)
	} else {
		if st.byHash == nil {
			st.byHash = make(map[uint64]cachedMultimodal)
		}
		c.bumpSessionEmbedLRULocked(sessionKey)
	}

	_, exists := st.byHash[hash]
	if !exists {
		maxHashes := c.sessionMaxHashes
		if maxHashes <= 0 {
			maxHashes = envconfig.ImageEmbedCacheMax()
		}
		for len(st.byHash) >= maxHashes && len(st.hashLRU) > 0 {
			victim := st.hashLRU[0]
			st.hashLRU = st.hashLRU[1:]
			delete(st.byHash, victim)
		}
	}
	// Share parts with global entry — session is a pin overlay, not a second copy.
	st.byHash[hash] = embed
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

func (c *VisionEmbedCache) bumpSessionEmbedLRULocked(key string) {
	for i, k := range c.sessionEmbedLRU {
		if k == key {
			c.sessionEmbedLRU = append(append(c.sessionEmbedLRU[:i], c.sessionEmbedLRU[i+1:]...), key)
			return
		}
	}
}

func (c *VisionEmbedCache) evictSessionEmbedLocked(key string) {
	delete(c.sessionEmbeds, key)
	for i, k := range c.sessionEmbedLRU {
		if k == key {
			c.sessionEmbedLRU = append(c.sessionEmbedLRU[:i], c.sessionEmbedLRU[i+1:]...)
			return
		}
	}
}
