package lfm2

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

func TestEffectiveFFDim(t *testing.T) {
	cfg := &Config{
		BlockFFDim:            6656,
		BlockAutoAdjustFFDim:  true,
		BlockFFNDimMultiplier: 1.0,
		BlockMultipleOf:       256,
	}
	got := effectiveFFDim(cfg)
	if got != 4608 {
		t.Fatalf("effectiveFFDim = %d, want 4608", got)
	}
}

func TestNewCachesLayout(t *testing.T) {
	m := &Model{
		Config: &Config{
			HiddenSize: 1024,
			ConvLCache: 3,
			FullAttnIdxs: []int{2, 5},
		},
		Layers: []*Layer{
			{IsAttention: false},
			{IsAttention: false},
			{IsAttention: true},
			{IsAttention: false},
			{IsAttention: false},
			{IsAttention: true},
		},
	}
	caches := m.NewCaches()
	if len(caches) != 6 {
		t.Fatalf("len=%d", len(caches))
	}
	if _, ok := caches[0].(*cache.RecurrentCache); !ok {
		t.Fatalf("cache[0]=%T", caches[0])
	}
	if _, ok := caches[2].(*cache.KVCache); !ok {
		t.Fatalf("cache[2]=%T", caches[2])
	}
	if _, ok := caches[5].(*cache.KVCache); !ok {
		t.Fatalf("cache[5]=%T", caches[5])
	}
}

func TestSanitizeConvWeightTranspose(t *testing.T) {
	// No MLX needed for nil path.
	if sanitizeConvWeight(nil) != nil {
		t.Fatal("nil in → nil out")
	}
}
