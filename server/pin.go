package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// pinLease is a session TTL hold that blocks eviction without loading models (Phase B3).
// WHY no load: Decide/Orient want to reserve residency intent without paying GetRunner cost
// or thrashing peers. WHY RuntimeGGUFs: protecting a ggml scheduler key alone does not stop
// Python ModelSwapGate from swapping; Go soft-pins the GGUF path and 503s conflicts.
type pinLease struct {
	ID           string
	Models       []string
	ModelKeys    []string
	RuntimeGGUFs []string // distinct runtime GGUF paths (at most one across host)
	ExpiresAt    time.Time
	ProjectID    string
	CoResident   bool
	Notes        string
}

func (g *mlxAgentGate) expirePinsLocked(now time.Time) {
	if g == nil || g.pins == nil {
		return
	}
	for id, lease := range g.pins {
		if lease == nil || !lease.ExpiresAt.After(now) {
			delete(g.pins, id)
		}
	}
}

func (g *mlxAgentGate) listPins() []api.PinStatus {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	now := time.Now()
	g.expirePinsLocked(now)
	out := make([]api.PinStatus, 0, len(g.pins))
	for _, lease := range g.pins {
		if lease == nil {
			continue
		}
		out = append(out, api.PinStatus{
			PinID:      lease.ID,
			Models:     append([]string(nil), lease.Models...),
			ExpiresAt:  lease.ExpiresAt.UTC(),
			ProjectID:  lease.ProjectID,
			CoResident: lease.CoResident,
			Notes:      lease.Notes,
		})
	}
	return out
}

func (g *mlxAgentGate) deletePin(id string) bool {
	if g == nil || id == "" {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.expirePinsLocked(time.Now())
	if _, ok := g.pins[id]; !ok {
		return false
	}
	delete(g.pins, id)
	return true
}

// pinUniqueKeyCountLocked returns distinct protected model keys across active pin leases.
// Caller must hold g.mu.
func (g *mlxAgentGate) pinUniqueKeyCountLocked() int {
	if g == nil || g.pins == nil {
		return 0
	}
	seen := make(map[string]struct{})
	for _, lease := range g.pins {
		if lease == nil {
			continue
		}
		for _, mk := range lease.ModelKeys {
			if mk != "" {
				seen[mk] = struct{}{}
			}
		}
	}
	return len(seen)
}

// pinWouldExceedBudgetLocked reports whether adding modelKeys would exceed maxN distinct keys.
// Caller must hold g.mu (and should have expirePinsLocked first).
// WHY global distinct keys: per-lease caps alone let N leases stack past MAX_LOADED.
func (g *mlxAgentGate) pinWouldExceedBudgetLocked(modelKeys []string, maxN uint) bool {
	if maxN == 0 {
		return false
	}
	seen := make(map[string]struct{})
	for _, lease := range g.pins {
		if lease == nil {
			continue
		}
		for _, mk := range lease.ModelKeys {
			if mk != "" {
				seen[mk] = struct{}{}
			}
		}
	}
	for _, mk := range modelKeys {
		if mk != "" {
			seen[mk] = struct{}{}
		}
	}
	return uint(len(seen)) > maxN
}

// pinnedRuntimeGGUFs returns active pin leases' runtime GGUF paths (may include duplicates).
func (g *mlxAgentGate) pinnedRuntimeGGUFs() []string {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.expirePinsLocked(time.Now())
	var out []string
	for _, lease := range g.pins {
		if lease == nil {
			continue
		}
		out = append(out, lease.RuntimeGGUFs...)
	}
	return out
}

// pinWouldConflictRuntimeGGUFLocked is true when adding ggufs would yield 2+ distinct runtime paths.
// WHY cross-lease: per-request fail-closed still allowed lease A=/tmp/a + lease B=/tmp/b.
func (g *mlxAgentGate) pinWouldConflictRuntimeGGUFLocked(ggufs []string) bool {
	var all []string
	for _, lease := range g.pins {
		if lease == nil {
			continue
		}
		all = append(all, lease.RuntimeGGUFs...)
	}
	all = append(all, ggufs...)
	return len(uniquePinnedRuntimeGGUFs(all)) > 1
}

func (g *mlxAgentGate) exclusiveFulfillmentActive() bool {
	if g == nil {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	hold, ok := g.fulfillmentActive(time.Now())
	return ok && hold.mode.Exclusive()
}

func newPinID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "pin_" + hex.EncodeToString(b[:])
}

type pinModelResolved struct {
	Name      string
	ModelKey  string
	Backend   string // ggml | runtime
	GGUF      string
	Model     *Model
}

func (s *Server) resolvePinModels(ctx context.Context, names []string) ([]pinModelResolved, error) {
	out := make([]pinModelResolved, 0, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		m, err := resolveCanLoadModel(name)
		if err != nil || m == nil {
			return nil, fmt.Errorf("model %q not found locally", name)
		}
		r := pinModelResolved{Name: name, Model: m, ModelKey: schedulerModelKey(m), Backend: "ggml"}
		if deferInferenceToRuntime(m) || modelUsesRuntimeInference(m) || envconfig.RuntimeProxyAll() {
			r.Backend = "runtime"
			opts := runtimeProxyOptions(name, 0, false, nil)
			if g, ok := opts["gguf"].(string); ok {
				r.GGUF = strings.TrimSpace(g)
			}
		}
		_ = ctx
		out = append(out, r)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("models is required")
	}
	return out, nil
}

func distinctRuntimeGGUFs(resolved []pinModelResolved) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, r := range resolved {
		if r.Backend != "runtime" {
			continue
		}
		key := r.GGUF
		if key == "" {
			key = "runtime:" + r.Name
		}
		if _, ok := seen[key]; ok {
			continue
		}
		// Path-normalize when both have paths
		dup := false
		for _, existing := range out {
			if ggufPathsEqual(existing, key) || existing == key {
				dup = true
				break
			}
		}
		if dup {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, key)
	}
	return out
}

func pinMaxAllowed(ctx context.Context) uint {
	if n := envconfig.PinMax(); n > 0 {
		return n
	}
	return effectiveMaxLoadedModels(ctx)
}

// PinHandler implements POST /api/pin — session lease; does not load models.
// WHY: stronger than request-scoped fulfillment for Orient/Decide session holds;
// fail closed on multi-runtime GGUF so clients never believe two Python models stay warm.
func (s *Server) PinHandler(c *gin.Context) {
	var req api.PinRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if len(req.Models) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "models is required"})
		return
	}
	ctx := c.Request.Context()
	resolved, err := s.resolvePinModels(ctx, req.Models)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if s != nil && s.sched != nil && s.sched.mlxGate.exclusiveFulfillmentActive() {
		c.JSON(http.StatusConflict, gin.H{
			"error": "exclusive fulfillment=benchmark is active; pin rejected",
			"notes": "wait for benchmark release or use a different host",
		})
		return
	}

	runtimeGGUFs := distinctRuntimeGGUFs(resolved)
	if len(runtimeGGUFs) > 1 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":       "cannot pin multiple distinct runtime GGUFs",
			"can_pin":     false,
			"co_resident": false,
			"notes":       "runtime_single_resident; serialize or pin one runtime model (Python holds one GGUF)",
		})
		return
	}

	maxN := pinMaxAllowed(ctx)
	if uint(len(resolved)) > maxN && maxN > 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   fmt.Sprintf("pin exceeds max (%d models)", maxN),
			"can_pin": false,
			"notes":   "ZEROLLAMA_PIN_MAX / OLLAMA_MAX_LOADED_MODELS",
		})
		return
	}

	ttl := envconfig.PinDefaultTTL()
	if req.TTLSeconds != nil && *req.TTLSeconds > 0 {
		ttl = time.Duration(*req.TTLSeconds) * time.Second
	}
	expires := time.Now().Add(ttl)
	coResident := len(runtimeGGUFs) <= 1
	notes := ""
	if len(runtimeGGUFs) == 1 && len(resolved) > 1 {
		notes = "runtime pin covers one GGUF (blocks other GGUFs); ggml keys also protected if present"
	}
	if coResident && len(runtimeGGUFs) == 0 {
		notes = "ggml multi-runner pin; eviction protect only (does not load)"
	}
	if coResident && len(runtimeGGUFs) == 1 && len(resolved) == 1 {
		notes = "runtime single-resident pin; blocks other GGUFs until expiry; ggml eviction protect for peers"
	}

	lease := &pinLease{
		ID:           newPinID(),
		Models:       make([]string, 0, len(resolved)),
		ModelKeys:    make([]string, 0, len(resolved)),
		RuntimeGGUFs: append([]string(nil), runtimeGGUFs...),
		ExpiresAt:    expires,
		ProjectID:    strings.TrimSpace(req.ProjectID),
		CoResident:   coResident,
		Notes:        notes,
	}
	for _, r := range resolved {
		lease.Models = append(lease.Models, r.Name)
		lease.ModelKeys = append(lease.ModelKeys, r.ModelKey)
	}

	if s == nil || s.sched == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "scheduler unavailable"})
		return
	}
	g := &s.sched.mlxGate
	g.mu.Lock()
	g.expirePinsLocked(time.Now())
	// Global pin budget: sum of distinct model keys across leases ≤ maxN.
	if g.pinWouldExceedBudgetLocked(lease.ModelKeys, maxN) {
		used := g.pinUniqueKeyCountLocked()
		g.mu.Unlock()
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   fmt.Sprintf("pin budget exceeded (%d keys in use, max %d)", used, maxN),
			"can_pin": false,
			"notes":   "ZEROLLAMA_PIN_MAX / OLLAMA_MAX_LOADED_MODELS; release pins or pin fewer models",
		})
		return
	}
	// Cross-lease: only one distinct runtime GGUF may be pinned host-wide.
	if g.pinWouldConflictRuntimeGGUFLocked(lease.RuntimeGGUFs) {
		g.mu.Unlock()
		c.JSON(http.StatusBadRequest, gin.H{
			"error":       "cannot pin a second distinct runtime GGUF while another is pinned",
			"can_pin":     false,
			"co_resident": false,
			"notes":       "runtime_single_resident; release the other pin or pin the same GGUF",
		})
		return
	}
	g.pins[lease.ID] = lease
	g.mu.Unlock()

	c.JSON(http.StatusOK, api.PinResponse{
		PinID:      lease.ID,
		ExpiresAt:  lease.ExpiresAt.UTC(),
		Models:     append([]string(nil), lease.Models...),
		ProjectID:  lease.ProjectID,
		CoResident: lease.CoResident,
		Notes:      lease.Notes,
		CanPin:     true,
	})
}

// UnpinHandler implements DELETE /api/pin/:id
func (s *Server) UnpinHandler(c *gin.Context) {
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "pin id required"})
		return
	}
	if s == nil || s.sched == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "scheduler unavailable"})
		return
	}
	if !s.sched.mlxGate.deletePin(id) {
		c.JSON(http.StatusNotFound, gin.H{"error": "pin not found or expired"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"ok": true, "pin_id": id})
}
