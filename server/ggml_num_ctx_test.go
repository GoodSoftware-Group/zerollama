package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

func TestGgmlKVQuantFallback(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_VRAM_MARGIN", "1.0")

	// Estimator that returns numCtx * 1000 bytes (f16) or numCtx * 500 (q8_0) or numCtx * 250 (q4_0).
	origEstimator := estimateGgmlLoadVRAM
	_ = origEstimator

	estimator := func(_ string, _ *ggml.GGML, numCtx int, profile ggmlLoadProfile) uint64 {
		switch profile.kvCacheType {
		case "q8_0":
			return uint64(numCtx * 500)
		case "q4_0":
			return uint64(numCtx * 250)
		default:
			return uint64(numCtx * 1000)
		}
	}

	f := &ggml.GGML{}
	profile := ggmlLoadProfile{batchSize: 512, numParallel: 1, kvCacheType: "f16"}
	// f16 costs 8_000_000; free = 5_000_000 → doesn't fit; q8_0 costs 4_000_000 → fits
	got := ggmlKVQuantFallbackWith(estimator, f, "m.gguf", 8000, profile, 5_000_000, 1.0)
	if got != "q8_0" {
		t.Fatalf("want q8_0, got %q", got)
	}

	// Both over budget; q4_0 fits at 2_000_000
	got = ggmlKVQuantFallbackWith(estimator, f, "m.gguf", 8000, profile, 3_000_000, 1.0)
	if got != "q4_0" {
		t.Fatalf("want q4_0, got %q", got)
	}

	// f16 already fits
	got = ggmlKVQuantFallbackWith(estimator, f, "m.gguf", 8000, profile, 10_000_000, 1.0)
	if got != "" {
		t.Fatalf("want empty (f16 fits), got %q", got)
	}
}

func TestCapGgmlNumCtx(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_CLAMP_NUM_CTX", "1")

	clamped, info := capGgmlNumCtx(262144, 8192, envconfig.GgmlClampNumCtxEnabled())
	if clamped != 8192 {
		t.Fatalf("clamped = %d", clamped)
	}
	if info == nil || !info.NumCtxClamped || info.NumCtxClampedFrom != 262144 {
		t.Fatalf("info = %#v", info)
	}

	unchanged, info := capGgmlNumCtx(4096, 8192, envconfig.GgmlClampNumCtxEnabled())
	if unchanged != 4096 {
		t.Fatalf("unchanged = %d", unchanged)
	}
	if info == nil || info.NumCtxClamped {
		t.Fatalf("expected suggest only, got %#v", info)
	}
}

func TestSuggestMaxGgmlNumCtxWithLinearEstimator(t *testing.T) {
	t.Setenv("ZEROLLAMA_GGML_VRAM_MARGIN", "1.0")

	estimator := func(_ string, _ *ggml.GGML, numCtx int, _ ggmlLoadProfile) uint64 {
		return uint64(numCtx * 1000)
	}
	f := &ggml.GGML{}
	profile := ggmlLoadProfile{batchSize: 512, numParallel: 1, kvCacheType: "f16"}

	got := suggestMaxGgmlNumCtxBounded(estimator, f, "model.gguf", 5_000_000, profile, 512, 8192)
	if got != 5000 {
		t.Fatalf("suggest = %d, want 5000", got)
	}
}

func TestEffectiveGgmlFreeVRAM(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")
	free := uint64(600 * format.MebiByte)
	gpus := []ml.DeviceInfo{{FreeMemory: free}}
	got := effectiveGgmlFreeVRAM(gpus)
	want := free - gpus[0].MinimumMemory()
	if got != want {
		t.Fatalf("free = %d, want %d", got, want)
	}
}

func TestApplyGgmlNumCtxResponseOnlyWhenClamped(t *testing.T) {
	res := &api.GenerateResponse{}
	applyGgmlNumCtxResponse(res, &api.GgmlNumCtx{SuggestedMaxNumCtx: 8192})
	if res.GgmlNumCtx != nil {
		t.Fatal("expected omit when not clamped")
	}
	applyGgmlNumCtxResponse(res, &api.GgmlNumCtx{NumCtxClamped: true, NumCtx: 8192})
	if res.GgmlNumCtx == nil || !res.GgmlNumCtx.NumCtxClamped {
		t.Fatal("expected clamped info on response")
	}
}

func TestCapNumCtxToModelMax(t *testing.T) {
	modelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":          "llama",
		"llama.context_length":          uint32(128_000),
		"llama.embedding_length":        uint32(4096),
		"llama.block_count":             uint32(1),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(32),
		"tokenizer.ggml.tokens":         []string{" "},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, nil)

	model := &Model{ModelPath: modelPath, ShortName: "test"}
	opts := api.DefaultOptions()
	opts.NumCtx = 262144
	info := capNumCtxToModelMax(model, &opts)
	if opts.NumCtx != 128_000 {
		t.Fatalf("num_ctx = %d, want 128000", opts.NumCtx)
	}
	if info == nil || !info.NumCtxClamped || info.NumCtxClampedFrom != 262144 || info.NumCtx != 128_000 {
		t.Fatalf("info = %#v", info)
	}

	opts = api.DefaultOptions()
	opts.NumCtx = 8192
	if capNumCtxToModelMax(model, &opts) != nil {
		t.Fatal("expected no cap below model max")
	}
}

func TestCapNumCtxToModelMaxMLX(t *testing.T) {
	model := &Model{
		ShortName: "mlx-test",
		Config: model.ConfigV2{
			ModelFormat: "safetensors",
			ContextLen:  32768,
		},
	}
	opts := api.DefaultOptions()
	opts.NumCtx = 262144
	info := capNumCtxToModelMax(model, &opts)
	if opts.NumCtx != 32768 {
		t.Fatalf("num_ctx = %d, want 32768", opts.NumCtx)
	}
	if info == nil || !info.NumCtxClamped {
		t.Fatalf("info = %#v", info)
	}
}

func TestMergeGgmlNumCtxInfo(t *testing.T) {
	train := &api.GgmlNumCtx{NumCtxClamped: true, NumCtxClampedFrom: 262144, NumCtx: 128_000}
	vram := &api.GgmlNumCtx{SuggestedMaxNumCtx: 8192, NumCtxClamped: true, NumCtxClampedFrom: 128_000, NumCtx: 8192}
	got := mergeGgmlNumCtxInfo(train, vram)
	if got.NumCtxClampedFrom != 262144 || got.NumCtx != 8192 || got.SuggestedMaxNumCtx != 8192 {
		t.Fatalf("merge = %#v", got)
	}
}

func TestEnrichShowGgmlNumCtxInfo(t *testing.T) {
	info := &api.GgmlNumCtx{SuggestedMaxNumCtx: 8192}
	got := enrichShowGgmlNumCtxInfo(4096, info)
	if got.MergedNumCtx != 0 {
		t.Fatalf("merged = %d, want 0", got.MergedNumCtx)
	}
	got = enrichShowGgmlNumCtxInfo(262144, info)
	if got.MergedNumCtx != 262144 {
		t.Fatalf("merged = %d, want 262144", got.MergedNumCtx)
	}
	if got.NumCtx != 0 {
		t.Fatalf("num_ctx should stay unset on show, got %d", got.NumCtx)
	}
}
