package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

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
