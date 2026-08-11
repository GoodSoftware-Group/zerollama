package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/runtimeclient"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
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

// CacheWarmHandler implements POST /api/cache/warm.
//
// WHY a dedicated endpoint instead of telling callers to send a throwaway
// /api/generate: warm is meant to run at boot (model/agent template preload)
// before real traffic exists. Routing it through generate would generate real
// tokens, show up in logs/metrics like a completion, and leave every caller to
// independently discover a "num_predict=1" convention.
//
// Routing:
//   - safetensors / MLX → scheduleRunner + Completion(num_predict=1) so the
//     prefix lands in the MLX trie under prompt_cache_key (same path as chat).
//   - GGUF → Python runtime /internal/cache/warm (L3 pinned slot / cache_prompt).
func (s *Server) CacheWarmHandler(c *gin.Context) {
	var req api.CacheWarmRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	key := strings.TrimSpace(req.PromptCacheKey)
	if key == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "prompt_cache_key is required"})
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "prompt is required"})
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found", req.Model)})
		return
	}
	m, err := GetModel(name.String())
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	var pinID string
	var expires *time.Time
	if req.Pin {
		ttl := envconfig.PinDefaultTTL()
		if req.TTLSeconds != nil && *req.TTLSeconds > 0 {
			ttl = time.Duration(*req.TTLSeconds) * time.Second
		}
		exp := time.Now().Add(ttl)
		lease := &cachePinLease{
			ID:             newCachePinID(),
			PromptCacheKey: key,
			ExpiresAt:      exp,
			Notes:          "created by /api/cache/warm",
		}
		cachePinMu.Lock()
		expireCachePinsLocked(time.Now())
		cachePinByID[lease.ID] = lease
		cachePinMu.Unlock()
		pinID = lease.ID
		expires = &exp
		runtimeclient.NotifyCachePin(c.Request.Context(), pinID, key, exp)
	}

	opts := map[string]any{}
	for k, v := range req.Options {
		opts[k] = v
	}
	opts["prompt_cache_key"] = key
	if req.NumCtx != nil && *req.NumCtx > 0 {
		opts["num_ctx"] = *req.NumCtx
	}
	opts["num_predict"] = 1

	keepAlive := &api.Duration{Duration: envconfig.KeepAlive()}
	if req.Pin {
		// Hold the runner long enough that a follow-up agent turn can hit the warm trie.
		keepAlive = &api.Duration{Duration: 30 * time.Minute}
	}

	isMLX := m.Config.ModelFormat == "safetensors" &&
		slices.Contains(m.Config.Capabilities, "completion")

	if isMLX {
		resp, status, warmErr := s.warmMLXCache(c.Request.Context(), name.String(), req.Prompt, key, opts, keepAlive)
		if warmErr != nil {
			c.JSON(status, gin.H{"error": warmErr.Error()})
			return
		}
		resp.PinID = pinID
		resp.ExpiresAt = expires
		c.JSON(http.StatusOK, resp)
		return
	}

	gguf, ok := runtimeGGUFPath(name.String())
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf(
				"model %q has no GGUF path for runtime L3 warm (format=%q); use an MLX safetensors completion model or a GGUF model",
				req.Model, m.Config.ModelFormat,
			),
		})
		return
	}

	result, err := runtimeclient.CacheWarm(
		c.Request.Context(), req.Prompt, key, gguf, req.NumCtx, pinID, expires, opts,
	)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, api.CacheWarmResponse{
		Warmed:         result.Warmed,
		PromptCacheKey: result.PromptCacheKey,
		KVDecodeSteps:  result.KVDecodeSteps,
		PinID:          pinID,
		ExpiresAt:      expires,
		Notes:          "GGUF prefix decoded into L3 runtime slot; no tokens were generated",
	})
}

func (s *Server) warmMLXCache(
	ctx context.Context,
	modelName, prompt, cacheKey string,
	opts map[string]any,
	keepAlive *api.Duration,
) (api.CacheWarmResponse, int, error) {
	runner, _, apiOpts, _, release, err := s.scheduleRunner(
		ctx, modelName, []model.Capability{model.CapabilityCompletion}, opts, keepAlive, nil, nil, nil,
	)
	if err != nil {
		return api.CacheWarmResponse{}, http.StatusInternalServerError, err
	}
	defer release()

	if apiOpts == nil {
		o := api.DefaultOptions()
		apiOpts = &o
	}
	apiOpts.NumPredict = 1

	var (
		cachedTokens int
		promptTokens int
	)
	err = runner.Completion(ctx, llm.CompletionRequest{
		Prompt:         prompt,
		Options:        apiOpts,
		PromptCacheKey: cacheKey,
		Shift:          true,
		Truncate:       true,
	}, func(cr llm.CompletionResponse) {
		if cr.Done {
			cachedTokens = cr.PromptEvalCachedCount
			promptTokens = cr.PromptEvalCount
		}
	})
	if err != nil {
		return api.CacheWarmResponse{}, http.StatusBadGateway, fmt.Errorf("mlx cache warm: %w", err)
	}

	slog.Info("mlx cache warm complete",
		"model", modelName,
		"prompt_cache_key", cacheKey,
		"prompt_tokens", promptTokens,
		"cached_tokens", cachedTokens,
	)
	return api.CacheWarmResponse{
		Warmed:         true,
		PromptCacheKey: cacheKey,
		Notes:          "MLX prefix prefilled into trie under prompt_cache_key; num_predict=1 discarded",
	}, http.StatusOK, nil
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
