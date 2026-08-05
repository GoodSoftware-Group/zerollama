// ggml_num_ctx — VRAM-aware suggest and opt-in clamp for merged manifest + request
// num_ctx on the Go ggml scheduler path (M12). Parity with Phase 13 runtime clamp
// (default off).
//
// Why this file exists: high-VRAM tier sets defaultNumCtx=262144; manifest parameters
// merge into modelOptions and pre-allocate full KV at llama.Load — qwen35/recurrent
// models can hang before the first token. Runtime path already exposes
// suggested_max_num_ctx; ggml needed the same operator signal without silent clamp.
package server

import (
	"context"
	"log/slog"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

const ggmlFreeVRAMCacheTTL = 2 * time.Second // Why: /api/show is called often; full GPUDevices refresh is ~500ms+.

// ggmlBootstrapProbe caches the result of the first nil-runner GPU discovery probe
// across all Server instances in the process. This prevents repeated 2s timeouts in
// test environments (where each test creates a fresh Server{}) and avoids duplicate
// discovery subprocesses in production.
var (
	ggmlBootstrapProbeMu   sync.Mutex
	ggmlBootstrapProbeAt   time.Time
	ggmlBootstrapProbeFree uint64
)

const ggmlBootstrapProbeCooldown = 10 * time.Minute

type ggmlLoadProfile struct {
	batchSize   int
	numParallel int
	kvCacheType string
	flash       ml.FlashAttentionType
}

func ggmlLoadProfileFor(model *Model, opts api.Options) ggmlLoadProfile {
	batch := opts.NumBatch
	if batch <= 0 {
		batch = api.DefaultOptions().NumBatch
	}
	if opts.NumCtx > 0 {
		batch = min(batch, opts.NumCtx)
	}

	numParallel := max(int(envconfig.NumParallel()), 1)
	if ggmlArchitectureForcesParallelOne(model) {
		numParallel = 1
	}

	kv := opts.KvCacheTypeEffective()

	return ggmlLoadProfile{
		batchSize:   batch,
		numParallel: numParallel,
		kvCacheType: kv,
		flash:       ml.FlashAttentionDisabled, // Why: FA lowers graph scratch; disabled = conservative suggest.
	}
}

func effectiveGgmlFreeVRAM(gpus []ml.DeviceInfo) uint64 {
	// Why same formula as sched.go load logging: GpuOverhead + MinimumMemory per device.
	var total uint64
	for _, gpu := range gpus {
		free := gpu.FreeMemory
		overhead := envconfig.GpuOverhead() + gpu.MinimumMemory()
		if free > overhead {
			free -= overhead
		} else {
			free = 0
		}
		total += free
	}
	return total
}

func estimateGgmlLoadVRAM(modelPath string, f *ggml.GGML, numCtx int, profile ggmlLoadProfile) uint64 {
	// Weights = file size (conservative proxy, same idea as PredictServerVRAM in llm).
	if numCtx <= 0 || f == nil {
		return 0
	}

	kv, partialOffload, fullOffload := f.GraphSize(
		uint64(numCtx),
		uint64(profile.batchSize),
		profile.numParallel,
		profile.kvCacheType,
		profile.flash,
	)

	var kvTotal uint64
	for _, k := range kv {
		kvTotal += k
	}

	// Use the sum of actual tensor byte sizes from GGUF metadata rather than the
	// file size. File size includes the GGUF header, KV metadata, and padding and
	// can exceed the actual tensor data by hundreds of MiB to several GiB for large
	// MoE models. Tensor.Size() returns exact quantized bytes per tensor, which is
	// what llama-server actually allocates into GPU buffers.
	// Fall back to file size only when metadata is unavailable.
	var weights uint64
	for _, t := range f.Tensors().Items() {
		weights += t.Size()
	}
	if weights == 0 {
		if info, err := os.Stat(modelPath); err == nil {
			weights = uint64(info.Size())
		}
	}

	graph := partialOffload
	if fullOffload > graph {
		graph = fullOffload
	}

	return weights + kvTotal + graph
}

// ggmlForceParallelOne families are not safe with llama-server -np > 1.
// Why qwen35/qwen35moe are absent: llama.cpp #20232 (Mar 2026) fixed the hybrid
// "Chunk not found" crash under parallel slots (#20222). Our pin (LLAMA_CPP_VERSION
// f95de977) includes that fix; keeping them here forced np=1 forever (stale vs
// ollama#17144). VL / lfm2 / nemotron_h* stay blocked pending their own verification.
var ggmlForceParallelOne = []string{
	"mllama", "qwen3vl", "qwen3vlmoe",
	"qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe",
}

func ggmlArchitectureForcesParallelOne(model *Model) bool {
	if model == nil {
		return false
	}
	family := model.PrimaryFamily()
	if family == "" {
		family = model.Config.ModelFamily
	}
	return slices.Contains(ggmlForceParallelOne, family)
}

// suggestMaxGgmlNumParallelWith returns the largest -np in [1, maxCap] whose load
// estimate fits in free VRAM (after margin). KV scales with num_ctx×np at load time.
func suggestMaxGgmlNumParallelWith(
	estimator ggmlVRAMEstimator,
	f *ggml.GGML,
	modelPath string,
	numCtx int,
	effectiveFree uint64,
	profile ggmlLoadProfile,
	maxCap int,
) int {
	if estimator == nil || f == nil || numCtx <= 0 || effectiveFree == 0 || maxCap < 1 {
		return 1
	}
	if maxCap > 64 {
		maxCap = 64
	}
	margin := envconfig.GgmlVRAMMargin()
	best := 1
	for np := 1; np <= maxCap; np++ {
		p := profile
		p.numParallel = np
		est := uint64(float64(estimator(modelPath, f, numCtx, p)) * margin)
		if est == 0 || est > effectiveFree {
			break
		}
		best = np
	}
	return best
}

func suggestMaxGgmlNumParallel(f *ggml.GGML, modelPath string, numCtx int, effectiveFree uint64, profile ggmlLoadProfile, maxCap int) int {
	return suggestMaxGgmlNumParallelWith(estimateGgmlLoadVRAM, f, modelPath, numCtx, effectiveFree, profile, maxCap)
}

// resolveGgmlNumParallel picks llama-server -np for this load.
// Auto (default): fit from free VRAM up to PARALLEL_MAX (and OLLAMA_NUM_PARALLEL when set).
// Auto off: OLLAMA_NUM_PARALLEL exactly. Go completion semaphore matches the chosen -np.
func resolveGgmlNumParallel(m *Model, opts api.Options, gpus []ml.DeviceInfo, f *ggml.GGML) int {
	requested := max(int(envconfig.NumParallel()), 1)
	if m != nil && m.CheckCapabilities(model.CapabilityCompletion) != nil {
		return 1
	}
	if ggmlArchitectureForcesParallelOne(m) {
		if requested != 1 {
			slog.Warn("model architecture does not currently support parallel requests",
				"architecture", m.PrimaryFamily(),
			)
		}
		return 1
	}
	if !envconfig.GgmlAutoParallelEnabled() || f == nil || m == nil || m.ModelPath == "" {
		return requested
	}

	maxCap := envconfig.GgmlParallelMaxCap()
	if envconfig.NumParallelExplicit() {
		maxCap = min(requested, maxCap)
	}
	numCtx := opts.NumCtx
	if numCtx <= 0 {
		numCtx = api.DefaultOptions().NumCtx
	}
	free := effectiveGgmlFreeVRAM(gpus)
	profile := ggmlLoadProfileFor(m, opts)
	fitted := suggestMaxGgmlNumParallel(f, m.ModelPath, numCtx, free, profile, maxCap)
	if fitted < 1 {
		fitted = 1
	}
	if fitted != requested {
		slog.Info("ggml num_parallel auto-fitted from VRAM",
			"model", m.ShortName,
			"num_ctx", numCtx,
			"requested", requested,
			"fitted", fitted,
			"cap", maxCap,
			"free_vram", format.HumanBytes2(free),
		)
	}
	return fitted
}

type ggmlVRAMEstimator func(modelPath string, f *ggml.GGML, numCtx int, profile ggmlLoadProfile) uint64

func suggestMaxGgmlNumCtxWith(
	estimator ggmlVRAMEstimator,
	f *ggml.GGML,
	modelPath string,
	effectiveFree uint64,
	profile ggmlLoadProfile,
) int {
	return suggestMaxGgmlNumCtxBounded(estimator, f, modelPath, effectiveFree, profile, 512, ggmlSuggestSearchHi(f))
}

func ggmlSuggestSearchHi(f *ggml.GGML) int {
	hi := envconfig.GgmlSuggestCtxMaxCap()
	if f == nil {
		return hi
	}
	train := f.KV().ContextLength()
	if train > 0 && int(train) < hi {
		return int(train)
	}
	return hi
}

func suggestMaxGgmlNumCtxBounded(
	estimator ggmlVRAMEstimator,
	f *ggml.GGML,
	modelPath string,
	effectiveFree uint64,
	profile ggmlLoadProfile,
	lo, hi int,
) int {
	if effectiveFree == 0 || hi < lo {
		return 0
	}
	margin := envconfig.GgmlVRAMMargin()

	required := func(ctx int) uint64 {
		return uint64(float64(estimator(modelPath, f, ctx, profile)) * margin)
	}

	if required(lo) > effectiveFree {
		return 0
	}
	if required(hi) <= effectiveFree {
		return hi
	}

	best := lo
	left, right := lo, hi
	for left <= right {
		mid := (left + right) / 2
		if required(mid) <= effectiveFree {
			best = mid
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return best
}

func suggestMaxGgmlNumCtx(f *ggml.GGML, modelPath string, effectiveFree uint64, profile ggmlLoadProfile) int {
	return suggestMaxGgmlNumCtxWith(estimateGgmlLoadVRAM, f, modelPath, effectiveFree, profile)
}

// modelMaxNumCtx returns n_ctx_train for GGUF models or manifest context_length for MLX.
func modelMaxNumCtx(model *Model) int {
	if model == nil {
		return 0
	}
	if model.IsMLX() {
		if model.Config.ContextLen > 0 {
			return model.Config.ContextLen
		}
		if model.Config.ModelFamily == "gemma4" {
			return 131072
		}
		return 0
	}
	if model.ModelPath == "" {
		return 0
	}
	f, err := llm.LoadModelMetadata(model.ModelPath)
	if err != nil {
		slog.Debug("model max num_ctx lookup skipped", "model", model.ShortName, "error", err)
		return 0
	}
	return int(f.KV().ContextLength())
}

// capNumCtxToModelMax lowers merged num_ctx to the model's trained/context limit before
// scheduler load. Clients may request (or inherit) a larger default; we cap silently.
func capNumCtxToModelMax(model *Model, opts *api.Options) *api.GgmlNumCtx {
	if opts == nil || model == nil || opts.NumCtx <= 0 {
		return nil
	}
	maxCtx := modelMaxNumCtx(model)
	if maxCtx <= 0 || opts.NumCtx <= maxCtx {
		return nil
	}
	from := opts.NumCtx
	opts.NumCtx = maxCtx
	slog.Info("num_ctx capped to model maximum",
		"model", model.ShortName,
		"from", from,
		"to", maxCtx,
	)
	return &api.GgmlNumCtx{
		NumCtxClamped:     true,
		NumCtxClampedFrom: from,
		NumCtx:            maxCtx,
	}
}

func mergeGgmlNumCtxInfo(train, vram *api.GgmlNumCtx) *api.GgmlNumCtx {
	if train == nil {
		return vram
	}
	if vram == nil {
		return train
	}
	out := *train
	if vram.SuggestedMaxNumCtx > 0 {
		out.SuggestedMaxNumCtx = vram.SuggestedMaxNumCtx
	}
	if vram.NumCtxClamped {
		out.NumCtx = vram.NumCtx
		// Preserve the original request when train cap ran first.
		if !train.NumCtxClamped {
			out.NumCtxClampedFrom = vram.NumCtxClampedFrom
		}
	}
	return &out
}

func capGgmlNumCtx(numCtx, suggested int, clampEnabled bool) (int, *api.GgmlNumCtx) {
	if numCtx <= 0 || suggested <= 0 {
		return numCtx, nil
	}
	info := &api.GgmlNumCtx{SuggestedMaxNumCtx: suggested}
	if !clampEnabled || numCtx <= suggested {
		return numCtx, info
	}
	info.NumCtxClamped = true
	info.NumCtxClampedFrom = numCtx
	info.NumCtx = suggested
	return suggested, info
}

func (s *Server) effectiveGgmlFreeVRAMForSuggest(ctx context.Context, refresh bool) uint64 {
	// refresh=true on load path: probe GPU drivers directly for current free bytes.
	// refresh=false on show: TTL cache avoids probe storm from CLI/UI show loops.
	//
	// Why we pass nil runners (not LoadedRunnersForDiscovery) on the refresh path:
	// Runner-derived FreeMemory is computed from log-parsed buffer sizes and a
	// "systemFreeAtLoad" baseline captured during model loading. For split-GPU models
	// (layer mode across multiple devices), this accounting is unreliable — the
	// systemFreeAtLoad baseline is set from "llama_prepare_model_devices" log lines
	// that fire at different phases (pre-weight and post-weight), and the min() of
	// accountedFree vs systemFree can severely undercount for dual-GPU layouts.
	// Passing nil forces a direct NVML/CUDA driver probe which is always accurate.
	if s == nil {
		return 0
	}
	if ctx == nil {
		ctx = context.Background()
	}

	s.ggmlFreeVRAMMu.Lock()
	cached, cachedAt := s.ggmlFreeVRAMCached, s.ggmlFreeVRAMAt
	s.ggmlFreeVRAMMu.Unlock()

	if !refresh {
		if cached > 0 && time.Since(cachedAt) < ggmlFreeVRAMCacheTTL {
			return cached
		}
	}

	// Determine runners for GPU device discovery.
	// With no loaded runners (first load), do a direct NVML/CUDA driver probe to
	// avoid relying on stale log-parsed accounting. Use a short timeout so test
	// environments without a real GPU library fail fast rather than blocking 30s.
	// The bootstrap probe result is cached at the process level (ggmlBootstrapProbe*)
	// so multiple Server instances (e.g. in tests) only pay the probe cost once.
	var runners []ml.FilteredRunnerDiscovery
	if s.sched != nil {
		runners = s.sched.LoadedRunnersForDiscovery()
	}

	var free uint64
	if len(runners) == 0 {
		// Check the process-level bootstrap cache before launching a subprocess.
		ggmlBootstrapProbeMu.Lock()
		bootstrapFree, bootstrapAt := ggmlBootstrapProbeFree, ggmlBootstrapProbeAt
		ggmlBootstrapProbeMu.Unlock()

		if bootstrapFree > 0 && time.Since(bootstrapAt) < ggmlFreeVRAMCacheTTL {
			// Bootstrap cached a GPU result; use it.
			free = bootstrapFree
		} else if bootstrapFree == 0 && time.Since(bootstrapAt) < ggmlBootstrapProbeCooldown {
			// Bootstrap recently failed; skip the probe to avoid repeated 2s waits.
			free = 0
		} else {
			// First probe or cooldown expired: launch the bootstrap subprocess with a
			// short timeout so test environments fail fast (2s vs the default 30s).
			probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
			gpus := discover.GPUDevices(probeCtx, nil)
			cancel()
			free = effectiveGgmlFreeVRAM(gpus)
			ggmlBootstrapProbeMu.Lock()
			ggmlBootstrapProbeAt = time.Now()
			ggmlBootstrapProbeFree = free
			ggmlBootstrapProbeMu.Unlock()
		}
	} else {
		// Runners are present: use runner-derived accounting directly (fast, no subprocess).
		gpus := discover.GPUDevices(ctx, runners)
		free = effectiveGgmlFreeVRAM(gpus)
	}

	if free > 0 {
		s.ggmlFreeVRAMMu.Lock()
		s.ggmlFreeVRAMCached = free
		s.ggmlFreeVRAMAt = time.Now()
		s.ggmlFreeVRAMMu.Unlock()
	}
	return free
}

func (s *Server) ggmlNumCtxSuggest(ctx context.Context, model *Model, opts api.Options, refreshFree bool) (int, *api.GgmlNumCtx) {
	if model == nil || model.ModelPath == "" || model.IsMLX() {
		return 0, nil
	}

	f, err := llm.LoadModelMetadata(model.ModelPath)
	if err != nil {
		slog.Debug("ggml num_ctx suggest skipped", "model", model.ShortName, "error", err)
		return 0, nil
	}

	free := s.effectiveGgmlFreeVRAMForSuggest(ctx, refreshFree)
	if free == 0 {
		return 0, nil
	}

	profile := ggmlLoadProfileFor(model, opts)
	suggested := suggestMaxGgmlNumCtx(f, model.ModelPath, free, profile)
	if suggested <= 0 {
		return 0, nil
	}
	return suggested, &api.GgmlNumCtx{SuggestedMaxNumCtx: suggested}
}

// ggmlKVQuantFallbackWith is the testable variant accepting a custom estimator.
func ggmlKVQuantFallbackWith(estimator ggmlVRAMEstimator, f *ggml.GGML, modelPath string, numCtx int, profile ggmlLoadProfile, free uint64, margin float64) string {
	// Only suggest a downgrade when f16 doesn't already fit.
	f16est := uint64(float64(estimator(modelPath, f, numCtx, profile)) * margin)
	if f16est <= free {
		return ""
	}
	for _, kt := range []string{"q8_0", "q4_0"} {
		p := profile
		p.kvCacheType = kt
		est := uint64(float64(estimator(modelPath, f, numCtx, p)) * margin)
		if est <= free {
			return kt
		}
	}
	return ""
}

// ggmlKVQuantFallback returns the cheapest KV cache type that would fit numCtx in free VRAM,
// trying q8_0 then q4_0 in order. Returns "" when f16 already fits or no fallback works.
func ggmlKVQuantFallback(f *ggml.GGML, modelPath string, numCtx int, profile ggmlLoadProfile, free uint64, margin float64) string {
	return ggmlKVQuantFallbackWith(estimateGgmlLoadVRAM, f, modelPath, numCtx, profile, free, margin)
}

// applyGgmlNumCtxClamp loads the GGUF metadata once, estimates memory for the requested
// context, warns loudly when it would exceed available memory, optionally suggests or
// applies a KV cache quantization fallback (request ggml_auto_kv_quant), and optionally
// clamps num_ctx to the VRAM-safe maximum (request ggml_clamp_num_ctx).
//
// Returns metadata for chat/generate responses; mutates opts in place.
func (s *Server) applyGgmlNumCtxClamp(ctx context.Context, model *Model, opts *api.Options) *api.GgmlNumCtx {
	if opts == nil || model == nil || model.ModelPath == "" || model.IsMLX() {
		return nil
	}

	f, err := llm.LoadModelMetadata(model.ModelPath)
	if err != nil {
		slog.Debug("ggml VRAM check skipped", "model", model.ShortName, "error", err)
		return nil
	}

	free := s.effectiveGgmlFreeVRAMForSuggest(ctx, true)
	if free == 0 {
		return nil
	}

	profile := ggmlLoadProfileFor(model, *opts)
	margin := envconfig.GgmlVRAMMargin()
	estimated := uint64(float64(estimateGgmlLoadVRAM(model.ModelPath, f, opts.NumCtx, profile)) * margin)

	out := &api.GgmlNumCtx{
		EstimatedLoadBytes: estimated,
		AvailableBytes:     free,
	}

	// Suggest VRAM-safe max regardless of clamping policy.
	suggested := suggestMaxGgmlNumCtx(f, model.ModelPath, free, profile)
	if suggested > 0 {
		out.SuggestedMaxNumCtx = suggested
	}

	if estimated > free {
		out.ExceedsAvailable = true

		// Check whether KV quantization would rescue the requested context.
		suggestedKV := ggmlKVQuantFallback(f, model.ModelPath, opts.NumCtx, profile, free, margin)
		if suggestedKV != "" {
			out.SuggestedKVCacheType = suggestedKV
		}

		slog.Warn("ggml load estimate exceeds available memory",
			"model", model.ShortName,
			"num_ctx", opts.NumCtx,
			"estimated", format.HumanBytes2(estimated),
			"available", format.HumanBytes2(free),
			"suggested_max_num_ctx", suggested,
			"suggested_kv_cache_type", suggestedKV,
		)

		// Auto KV-quant: downgrade KV cache type when this request opts in.
		if suggestedKV != "" && opts.GgmlAutoKVQuantEnabled() {
			fromKV := profile.kvCacheType
			if fromKV == "" {
				fromKV = "f16"
			}
			opts.KvCacheType = suggestedKV
			profile.kvCacheType = suggestedKV
			estimated = uint64(float64(estimateGgmlLoadVRAM(model.ModelPath, f, opts.NumCtx, profile)) * margin)
			out.EstimatedLoadBytes = estimated
			out.ExceedsAvailable = estimated > free
			out.KVCacheTypeDowngraded = true
			out.KVCacheTypeFrom = fromKV
			out.KVCacheType = suggestedKV
			slog.Warn("ggml KV cache type auto-downgraded for VRAM",
				"model", model.ShortName,
				"from", fromKV,
				"to", suggestedKV,
			)
		}
	}

	// VRAM-based context clamp (request ggml_clamp_num_ctx or ZEROLLAMA_GGML_CLAMP_NUM_CTX).
	if suggested > 0 {
		clamp := opts.GgmlClampNumCtxEnabled() || envconfig.GgmlClampNumCtxEnabled()
		clamped, clampInfo := capGgmlNumCtx(opts.NumCtx, suggested, clamp)
		if clampInfo != nil {
			out.NumCtxClamped = clampInfo.NumCtxClamped
			out.NumCtxClampedFrom = clampInfo.NumCtxClampedFrom
			out.NumCtx = clampInfo.NumCtx
		}
		if clamped != opts.NumCtx {
			slog.Warn("ggml num_ctx clamped for VRAM",
				"model", model.ShortName,
				"from", opts.NumCtx,
				"to", clamped,
				"suggested_max", suggested,
			)
			opts.NumCtx = clamped
		}
	}

	return out
}

func applyGgmlNumCtxResponse(res *api.GenerateResponse, info *api.GgmlNumCtx) {
	// Emit when clamped OR when load exceeds memory — operators need the signal either way.
	if res == nil || info == nil || (!info.NumCtxClamped && !info.ExceedsAvailable && !info.KVCacheTypeDowngraded) {
		return
	}
	res.GgmlNumCtx = info
}

func applyGgmlNumCtxChatResponse(res *api.ChatResponse, info *api.GgmlNumCtx) {
	// Emit when clamped OR when load exceeds memory — operators need the signal either way.
	if res == nil || info == nil || (!info.NumCtxClamped && !info.ExceedsAvailable && !info.KVCacheTypeDowngraded) {
		return
	}
	res.GgmlNumCtx = info
}

func enrichShowGgmlNumCtxInfo(mergedNumCtx int, info *api.GgmlNumCtx) *api.GgmlNumCtx {
	// Why MergedNumCtx not NumCtx: NumCtx means effective after clamp on load responses;
	// show must not reuse that field for "you asked for this but it exceeds VRAM".
	if info == nil {
		return nil
	}
	if mergedNumCtx > info.SuggestedMaxNumCtx {
		info.MergedNumCtx = mergedNumCtx
	}
	return info
}

func (s *Server) enrichShowGgmlNumCtx(ctx context.Context, resp *api.ShowResponse, m *Model) {
	if resp == nil || m == nil {
		return
	}
	opts, err := s.modelOptions(m, nil)
	if err != nil {
		return
	}
	_, info := s.ggmlNumCtxSuggest(ctx, m, opts, false)
	resp.GgmlNumCtx = enrichShowGgmlNumCtxInfo(opts.NumCtx, info)
}
