package server

import (
	"context"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/runtimeclient"
)

const (
	// exact: runtime ProbeVramEstimate (real budget). heuristic: ggml count/group only.
	// Why two levels: Decide must not treat ggml guesses as hard VRAM truth.
	canLoadConfidenceExact     = "exact"
	canLoadConfidenceHeuristic = "heuristic"
)

// CanLoadHandler implements POST /api/can-load — capacity dry-run without starting inference.
//
// Why: Orient/Decide need admit budgets before a graph run. Calling GetRunner would
// enqueue real loads (and can itself hit ErrMaxQueue), so this path never loads.
// Always HTTP 200 with structured fields — busy is a field, not a 503 (probes ≠ traffic).
func (s *Server) CanLoadHandler(c *gin.Context) {
	var req api.CanLoadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	c.JSON(http.StatusOK, s.evaluateCanLoad(c.Request.Context(), req))
}

func (s *Server) evaluateCanLoad(ctx context.Context, req api.CanLoadRequest) api.CanLoadResponse {
	maxLoaded := effectiveMaxLoadedModels(ctx)
	maxQueue := envconfig.MaxQueue()
	resp := api.CanLoadResponse{
		Model:           req.Model,
		Confidence:      canLoadConfidenceHeuristic,
		MaxLoadedModels: maxLoaded,
		Queue: api.CanLoadQueueSnapshot{
			GgmlMaxQueue: maxQueue,
		},
	}

	var warm api.ProcessResponse
	loadsPaused := false
	ggmlPending := 0
	loadedCount := 0
	if s != nil && s.sched != nil {
		warm = s.sched.ProcessSnapshot()
		snap := s.sched.InferenceFleetSnapshot()
		loadsPaused = snap.LoadsPaused
		ggmlPending = snap.Pending
		loadedCount = snap.Loaded
	}
	resp.Warm = warm
	resp.LoadsPaused = loadsPaused
	resp.LoadedCount = loadedCount
	resp.Queue.GgmlPending = ggmlPending

	runtimeWaiting := 0
	runtimeMaxQ := runtimeMaxQueueFromEnv()
	var runtimeHealth runtimeHealthSnapshot
	if runtimeHealthProbeRequired() {
		runtimeHealth = runtimeInferenceHealth(ctx)
		if runtimeHealth.ok {
			runtimeWaiting = runtimeHealth.waiting
			resp.Queue.RuntimeWaiting = runtimeWaiting
			resp.Queue.RuntimeMaxQueue = runtimeMaxQ
		}
	}

	if loadsPaused {
		resp.Busy = true
		resp.BusyReason = "loads_paused"
	} else if uint(ggmlPending) >= maxQueue && maxQueue > 0 {
		resp.Busy = true
		resp.BusyReason = "ggml_max_queue"
	} else if runtimeHealthProbeRequired() && runtimeHealth.ok && uint(runtimeWaiting) >= runtimeMaxQ && runtimeMaxQ > 0 {
		resp.Busy = true
		resp.BusyReason = "runtime_max_queue"
	}

	m, modelErr := resolveCanLoadModel(req.Model)
	useRuntime := false
	if modelErr == nil && m != nil {
		useRuntime = deferInferenceToRuntime(m) || modelUsesRuntimeInference(m)
	}
	// Explicit gguf option or global runtime proxy → treat as runtime path for estimate.
	if !useRuntime && effectiveRuntimeURL() != "" {
		if g, ok := req.Options["gguf"].(string); ok && strings.TrimSpace(g) != "" {
			useRuntime = true
		} else if envconfig.RuntimeProxyAll() {
			useRuntime = true
		}
	}

	if useRuntime {
		resp.Backend = "runtime"
		resp.Confidence = canLoadConfidenceExact
		return s.evaluateCanLoadRuntime(ctx, req, m, resp, runtimeHealth)
	}

	resp.Backend = "ggml"
	resp.Notes = "ggml can_load is heuristic (count/concurrency groups only); does not guarantee VRAM fit. Treat needs_eviction as thrash risk."
	return evaluateCanLoadGgml(s, req, m, resp)
}

func resolveCanLoadModel(modelName string) (*Model, error) {
	modelRef, err := parseAndValidateModelRef(modelName)
	if err != nil {
		return nil, err
	}
	if modelRef.Source == modelSourceCloud {
		return nil, errModelNotLocal
	}
	name, err := getExistingName(modelRef.Name)
	if err != nil {
		return nil, err
	}
	return GetModel(name.String())
}

var errModelNotLocal = errCanLoadNotLocal{}

type errCanLoadNotLocal struct{}

func (errCanLoadNotLocal) Error() string { return "cloud model; can-load is local-only" }

// modelAlreadyWarm matches the requested tag against ggml /api/ps residents.
// Why: "any runner warm" is wrong for single-resident runtime — see ggufPathsEqual.
func modelAlreadyWarm(warm api.ProcessResponse, modelName string, m *Model) bool {
	want := strings.ToLower(strings.TrimSpace(modelName))
	short := ""
	if m != nil {
		short = strings.ToLower(strings.TrimSpace(m.ShortName))
		if short == "" {
			short = strings.ToLower(strings.TrimSpace(m.Name))
		}
	}
	for _, pm := range warm.Models {
		n := strings.ToLower(pm.Name)
		mod := strings.ToLower(pm.Model)
		if n == want || mod == want || (short != "" && (n == short || mod == short)) {
			return true
		}
	}
	return false
}

// ggufPathsEqual compares resolved paths so already_loaded is GGUF-accurate.
// Why: Python ModelSwapGate holds one GGUF; matching only "llama loaded" lied to Decide.
func ggufPathsEqual(a, b string) bool {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == "" || b == "" {
		return false
	}
	ap, errA := filepath.EvalSymlinks(a)
	bp, errB := filepath.EvalSymlinks(b)
	if errA != nil {
		ap = a
	}
	if errB != nil {
		bp = b
	}
	return filepath.Clean(ap) == filepath.Clean(bp)
}

func evaluateCanLoadGgml(s *Server, req api.CanLoadRequest, m *Model, resp api.CanLoadResponse) api.CanLoadResponse {
	if modelAlreadyWarm(resp.Warm, req.Model, m) {
		resp.AlreadyLoaded = true
		resp.CanLoad = !resp.Busy
		return resp
	}
	if m == nil {
		resp.CanLoad = false
		resp.Notes = "model not found locally; " + resp.Notes
		return resp
	}

	needsEvict := false
	reason := ""
	if s != nil && s.sched != nil {
		if conflict := s.sched.findConcurrencyGroupConflict(m); conflict != nil {
			needsEvict = true
			reason = "concurrency_group"
		}
	}
	if !needsEvict && resp.MaxLoadedModels > 0 && uint(resp.LoadedCount) >= resp.MaxLoadedModels {
		needsEvict = true
		reason = "max_loaded_models"
	}
	resp.NeedsEviction = needsEvict
	resp.EvictionReason = reason
	// can_load and needs_eviction are orthogonal: scheduler may admit by unloading
	// a victim. Why: opportunistic work can thrash; hire maps must require !needs_eviction.
	resp.CanLoad = !resp.Busy
	if needsEvict && resp.CanLoad {
		resp.Notes = strings.TrimSpace(resp.Notes + " needs_eviction=true: load would thrash another resident model")
	}
	return resp
}

func (s *Server) evaluateCanLoadRuntime(
	ctx context.Context,
	req api.CanLoadRequest,
	m *Model,
	resp api.CanLoadResponse,
	h runtimeHealthSnapshot,
) api.CanLoadResponse {
	opts := runtimeProxyOptions(req.Model, 0, false, req.Options)
	gguf, _ := opts["gguf"].(string)
	gguf = strings.TrimSpace(gguf)

	// Warm only when the requested GGUF matches the runtime resident path,
	// or the ggml /api/ps warm set lists this model name.
	if modelAlreadyWarm(resp.Warm, req.Model, m) {
		resp.AlreadyLoaded = true
	} else if h.ok && h.llamaLoaded && gguf != "" && ggufPathsEqual(gguf, h.loadedGGUF) {
		resp.AlreadyLoaded = true
	}

	if gguf == "" {
		// Fail closed: soft-admit without a path caused thrash when Decide trusted can_load.
		resp.Confidence = canLoadConfidenceHeuristic
		resp.CanLoad = false
		resp.Notes = "runtime path but no gguf resolved; refuse admit (fail closed)"
		return resp
	}

	snap := runtimeclient.ProbeVramEstimate(ctx, gguf, opts)
	if snap == nil {
		// Fail closed: unavailable estimate ≠ "probably fine". Prefer unknown/no over false yes.
		resp.Confidence = canLoadConfidenceHeuristic
		resp.CanLoad = false
		resp.Notes = "runtime vram-estimate unavailable; refuse admit (fail closed)"
		return resp
	}
	if est, ok := snap["vram_estimate"].(map[string]any); ok {
		resp.VramEstimate = est
	}
	if budget, ok := snap["vram_budget"].(map[string]any); ok {
		resp.VramBudget = budget
		if v, ok := budget["suggested_max_num_ctx"].(float64); ok {
			n := int(v)
			resp.SuggestedMaxNumCtx = &n
		} else if v, ok := budget["suggested_max_num_ctx"].(int); ok {
			resp.SuggestedMaxNumCtx = &v
		}
		fits := true
		if v, ok := budget["fits_with_margin"].(bool); ok {
			fits = v
		} else if v, ok := budget["fits"].(bool); ok {
			fits = v
		}
		if host, ok := budget["host_ram"].(map[string]any); ok {
			if hf, ok := host["fits"].(bool); ok && !hf {
				fits = false
			}
		}
		if !fits {
			if resp.AlreadyLoaded {
				// Already the resident GGUF — can run without a new load.
				resp.CanLoad = !resp.Busy
			} else if h.ok && h.llamaLoaded {
				// Runtime is single-resident: another GGUF would force a swap.
				resp.NeedsEviction = true
				resp.EvictionReason = "vram_budget"
				resp.CanLoad = !resp.Busy
				resp.Notes = "needs_eviction=true: VRAM does not fit alongside current runtime resident; swap required"
			} else if resp.LoadedCount > 0 {
				resp.NeedsEviction = true
				resp.EvictionReason = "vram_budget"
				resp.CanLoad = !resp.Busy
				resp.Notes = "needs_eviction=true: VRAM does not fit; ggml residents may need unload"
			} else {
				resp.CanLoad = false
				resp.Notes = "vram_budget does not fit with margin and nothing to evict"
			}
			return resp
		}
	}
	resp.CanLoad = !resp.Busy
	if resp.NeedsEviction && resp.CanLoad {
		resp.Notes = strings.TrimSpace(resp.Notes + " needs_eviction=true: load would thrash another resident model")
	}
	return resp
}
