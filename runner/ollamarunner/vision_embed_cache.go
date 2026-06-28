package ollamarunner

import (
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
)

var errVisionEmbedNotFound = errors.New("vision embed not found in cache")

type sessionEmbedState struct {
	byHash    map[uint64]cachedMultimodal
	updatedAt time.Time
}

type imageEmbedCache struct {
	key      uint64
	val      cachedMultimodal
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

// VisionEmbedCache stores materialized vision encoder outputs (float32 + metadata)
// so agent threads can skip re-encoding the same clip frames between turns.
type VisionEmbedCache struct {
	mu sync.Mutex

	entries   []imageEmbedCache
	imageHash maphash.Hash

	sessionEmbeds   map[string]sessionEmbedState
	sessionEmbedLRU []string
}

func NewVisionEmbedCache(cacheSize int) *VisionEmbedCache {
	if cacheSize <= 0 {
		cacheSize = defaultImageCacheSize
	}
	return &VisionEmbedCache{
		entries: make([]imageEmbedCache, cacheSize),
	}
}

func normalizeSessionKey(sessionKey string) string {
	return strings.TrimSpace(sessionKey)
}

func (c *VisionEmbedCache) hashImage(image []byte) uint64 {
	c.imageHash.Reset()
	_, _ = c.imageHash.Write(image)
	return c.imageHash.Sum64()
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
	sessionKey string,
	sessionOverlay bool,
) ([]input.Multimodal, error) {
	if c == nil {
		return mp.EncodeMultimodal(ctx, data)
	}
	if len(data) == 0 {
		return nil, errors.New("received zero length image")
	}

	hash := c.hashImage(data)
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
		slog.Info("vision embed global cache hit", "engine", "ollama")
		return restoreMultimodal(ctx, cached), nil
	}
	c.mu.Unlock()

	encodeCtx := backend.NewContext()
	mm, err := mp.EncodeMultimodal(encodeCtx, data)
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

func (c *VisionEmbedCache) addGlobalLocked(hash uint64, embed cachedMultimodal) {
	best := time.Now()
	bestEntry := 0

	for i := range c.entries {
		if c.entries[i].key == hash {
			bestEntry = i
			break
		}
		if c.entries[i].lastUsed.Compare(best) < 0 {
			best = c.entries[i].lastUsed
			bestEntry = i
		}
	}

	slog.Debug("storing image embeddings in cache",
		"entry", bestEntry,
		"used", c.entries[bestEntry].lastUsed,
		"engine", "ollama",
	)
	c.entries[bestEntry].key = hash
	c.entries[bestEntry].val = embed
	c.entries[bestEntry].lastUsed = time.Now()
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
	st.byHash[hash] = embed
	st.updatedAt = time.Now()
	c.sessionEmbeds[sessionKey] = st
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
