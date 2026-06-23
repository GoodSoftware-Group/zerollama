package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestEffectiveChatPromptBudgetCapsToModelMax(t *testing.T) {
	m := &Model{
		Config: model.ConfigV2{ContextLen: 131_072},
	}
	opts := &api.Options{
		Runner:     api.Runner{NumCtx: 262_144},
		NumPredict: 65_536,
	}
	got := effectiveChatPromptBudget(opts, m, 131_072)
	want := renderPromptTokenBudget(131_072, 65_536)
	if got != want {
		t.Fatalf("budget = %d want %d", got, want)
	}
	if got >= 262_144 {
		t.Fatalf("budget should be capped below inflated num_ctx, got %d", got)
	}
}

func TestMLXKeepAliveFloor(t *testing.T) {
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}}
	gguf := &Model{Config: model.ConfigV2{ModelFormat: "gguf"}}

	// nil model or non-MLX: pass through unchanged
	if got := mlxKeepAliveFloor(nil, nil); got != nil {
		t.Fatalf("nil model: want nil, got %v", got)
	}
	if got := mlxKeepAliveFloor(gguf, nil); got != nil {
		t.Fatalf("gguf: want nil, got %v", got)
	}

	// explicit keep-alive respected
	explicit := &api.Duration{Duration: 10 * time.Second}
	if got := mlxKeepAliveFloor(mlx, explicit); got != explicit {
		t.Fatalf("explicit: want %v, got %v", explicit, got)
	}

	// nil keep-alive on MLX: should return at least mlxMinKeepAlive
	got := mlxKeepAliveFloor(mlx, nil)
	if got == nil {
		t.Fatal("mlx nil: want non-nil floor")
	}
	if got.Duration < mlxMinKeepAlive {
		t.Fatalf("mlx nil: got %v, want >= %v", got.Duration, mlxMinKeepAlive)
	}
}

func TestCapMLXScheduleOptions(t *testing.T) {
	m := &Model{
		Config: model.ConfigV2{
			ModelFormat: "safetensors",
			ContextLen:  131_072,
			ModelFamily: "gemma4",
		},
	}
	opts := api.Options{Runner: api.Runner{NumCtx: 262_144}}
	capMLXScheduleOptions(m, &opts)
	if opts.NumCtx != 131_072 {
		t.Fatalf("num_ctx = %d want 131072", opts.NumCtx)
	}
}
