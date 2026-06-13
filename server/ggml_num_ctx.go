// ggml_num_ctx: VRAM-aware suggest/clamp for merged manifest + request num_ctx on the
// Go ggml scheduler path. Parity with Phase 13 runtime clamp (default off).
package server

import (
	"log/slog"
	"os"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

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
	family := model.PrimaryFamily()
	if family == "" {
		family = model.Config.ModelFamily
	}
	if slices.Contains([]string{
		"mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe",
		"qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe",
	}, family) {
		numParallel = 1
	}

	kv := envconfig.KvCacheType()
	if kv == "" {
		kv = "f16"
	}

	return ggmlLoadProfile{
		batchSize:   batch,
		numParallel: numParallel,
		kvCacheType: kv,
		flash:       ml.FlashAttentionDisabled, // conservative for suggest
	}
}

func effectiveGgmlFreeVRAM(gpus []ml.DeviceInfo) uint64 {
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

	var weights uint64
	if info, err := os.Stat(modelPath); err == nil {
		weights = uint64(info.Size())
	}

	graph := partialOffload
	if fullOffload > graph {
		graph = fullOffload
	}

	return weights + kvTotal + graph
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
	defer func() {
		_ = recover()
	}()
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

func (s *Server) ggmlNumCtxSuggest(model *Model, opts api.Options) (int, *api.GgmlNumCtx) {
	if model == nil || model.ModelPath == "" || model.IsMLX() {
		return 0, nil
	}

	f, err := llm.LoadModel(model.ModelPath, 1024)
	if err != nil {
		slog.Debug("ggml num_ctx suggest skipped", "model", model.ShortName, "error", err)
		return 0, nil
	}

	gpus := discover.GPUDevices(nil, nil)
	free := effectiveGgmlFreeVRAM(gpus)
	if free == 0 && s.totalVRAM > 0 {
		// Discovery at load time may be fresher than startup; fall back to tier total.
		free = s.totalVRAM
	}
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

// applyGgmlNumCtxClamp optionally lowers merged num_ctx before ggml load (env-gated).
func (s *Server) applyGgmlNumCtxClamp(model *Model, opts *api.Options) *api.GgmlNumCtx {
	if opts == nil {
		return nil
	}
	suggested, info := s.ggmlNumCtxSuggest(model, *opts)
	if suggested <= 0 {
		return nil
	}
	clamped, out := capGgmlNumCtx(opts.NumCtx, suggested, envconfig.GgmlClampNumCtxEnabled())
	if out == nil {
		return info
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
	return out
}

func applyGgmlNumCtxResponse(res *api.GenerateResponse, info *api.GgmlNumCtx) {
	if res == nil || info == nil || !info.NumCtxClamped {
		return
	}
	res.GgmlNumCtx = info
}

func applyGgmlNumCtxChatResponse(res *api.ChatResponse, info *api.GgmlNumCtx) {
	if res == nil || info == nil || !info.NumCtxClamped {
		return
	}
	res.GgmlNumCtx = info
}

func (s *Server) enrichShowGgmlNumCtx(resp *api.ShowResponse, m *Model) {
	if resp == nil || m == nil {
		return
	}
	opts, err := s.modelOptions(m, nil)
	if err != nil {
		return
	}
	_, info := s.ggmlNumCtxSuggest(m, opts)
	if info == nil {
		return
	}
	if opts.NumCtx > info.SuggestedMaxNumCtx {
		info.NumCtx = opts.NumCtx
	}
	resp.GgmlNumCtx = info
}
