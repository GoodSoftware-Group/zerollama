package server

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/runtimeclient"
	"github.com/ollama/ollama/x/mlxrunner"
)

// cachePinLease holds a TTL pin on a prompt_cache_key (not a model name).
// WHY separate from /api/pin: model residency ≠ prefix-cache residency. Hermes
// wants L3/MLX trie branches to survive idle gaps without forcing a model pin.
//
// Limitation (documented): does NOT force llama-server id_slot retention while
// no request is in flight — SlotAllocator does not hold idle slots today.
type cachePinLease struct {
	ID             string
	PromptCacheKey string
	ExpiresAt      time.Time
	ProjectID      string
	Notes          string
}

var (
	cachePinMu   sync.Mutex
	cachePinByID = map[string]*cachePinLease{}
)

func expireCachePinsLocked(now time.Time) {
	for id, lease := range cachePinByID {
		if lease == nil || !lease.ExpiresAt.After(now) {
			delete(cachePinByID, id)
		}
	}
}

func cacheKeyIsPinned(key string) bool {
	key = strings.TrimSpace(key)
	if key == "" {
		return false
	}
	cachePinMu.Lock()
	defer cachePinMu.Unlock()
	expireCachePinsLocked(time.Now())
	for _, lease := range cachePinByID {
		if lease != nil && lease.PromptCacheKey == key {
			return true
		}
	}
	return false
}

func init() {
	// WHY package hook: mlxrunner cannot import server (cycle); server registers
	// the pin check so trie eviction skips pinned prompt_cache_key branches.
	mlxrunner.CacheKeyPinned = cacheKeyIsPinned
}

func newCachePinID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "cpin_" + hex.EncodeToString(b[:])
}

// CachePinHandler implements POST /api/cache/pin.
func (s *Server) CachePinHandler(c *gin.Context) {
	var req api.CachePinRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	key := strings.TrimSpace(req.PromptCacheKey)
	if key == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "prompt_cache_key is required"})
		return
	}
	ttl := envconfig.PinDefaultTTL()
	if req.TTLSeconds != nil && *req.TTLSeconds > 0 {
		ttl = time.Duration(*req.TTLSeconds) * time.Second
	}
	expires := time.Now().Add(ttl)
	lease := &cachePinLease{
		ID:             newCachePinID(),
		PromptCacheKey: key,
		ExpiresAt:      expires,
		ProjectID:      strings.TrimSpace(req.ProjectID),
		Notes:          "pins MLX trie + L3 disk TTL for this key; does not retain idle llama-server slots",
	}

	cachePinMu.Lock()
	expireCachePinsLocked(time.Now())
	cachePinByID[lease.ID] = lease
	cachePinMu.Unlock()

	// Best-effort notify Python so L3 eviction uses extended TTL for this key.
	runtimeclient.NotifyCachePin(c.Request.Context(), lease.ID, key, expires)

	c.JSON(http.StatusOK, api.CachePinResponse{
		PinID:          lease.ID,
		PromptCacheKey: lease.PromptCacheKey,
		ExpiresAt:      lease.ExpiresAt.UTC(),
		ProjectID:      lease.ProjectID,
		Notes:          lease.Notes,
		CanPin:         true,
	})
}

// CacheUnpinHandler implements DELETE /api/cache/pin/:id.
func (s *Server) CacheUnpinHandler(c *gin.Context) {
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "pin id required"})
		return
	}
	cachePinMu.Lock()
	expireCachePinsLocked(time.Now())
	lease, ok := cachePinByID[id]
	if ok {
		delete(cachePinByID, id)
	}
	cachePinMu.Unlock()
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "cache pin not found"})
		return
	}
	runtimeclient.NotifyCacheUnpin(c.Request.Context(), id, lease.PromptCacheKey)
	c.JSON(http.StatusOK, gin.H{"deleted": true, "pin_id": id})
}
