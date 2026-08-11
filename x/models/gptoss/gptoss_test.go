package gptoss

import (
	"context"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestParseConfigGPTOSS120B(t *testing.T) {
	data := []byte(`{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"hidden_size": 2880,
		"intermediate_size": 2880,
		"num_hidden_layers": 36,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"head_dim": 64,
		"num_local_experts": 128,
		"num_experts_per_tok": 4,
		"sliding_window": 128,
		"max_position_embeddings": 131072,
		"initial_context_length": 4096,
		"rope_theta": 150000,
		"rope_scaling": {"factor": 32.0}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.NumHiddenLayers != 36 {
		t.Errorf("NumHiddenLayers = %d, want 36", cfg.NumHiddenLayers)
	}
	if cfg.NumLocalExperts != 128 {
		t.Errorf("NumLocalExperts = %d, want 128", cfg.NumLocalExperts)
	}
	if cfg.NumExpertsPerTok != 4 {
		t.Errorf("NumExpertsPerTok = %d, want 4", cfg.NumExpertsPerTok)
	}
	if cfg.RopeScalingFactor != 32 {
		t.Errorf("RopeScalingFactor = %v, want 32", cfg.RopeScalingFactor)
	}
	if cfg.RopeInvScale != 1.0/32.0 {
		t.Errorf("RopeInvScale = %v, want %v", cfg.RopeInvScale, 1.0/32.0)
	}
	if len(cfg.LayerTypes) != 36 {
		t.Fatalf("layer_types len = %d, want 36 (auto-filled)", len(cfg.LayerTypes))
	}
	if !cfg.layerIsSliding(0) || cfg.layerIsSliding(1) {
		t.Errorf("unexpected sliding pattern at layers 0/1")
	}
}

// Regression: Metal argpartition on rank-3 router logits returns ndim=0, which
// made SparseMoE.route panic on shape[0]. Prefer the flattened [BL, E] path.
func TestArgpartitionTopK2D(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	thread, err := mlxthread.Start("gptoss-argpartition", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX thread: %v", err)
	}
	defer func() {
		_ = thread.Stop(context.Background(), func() {
			mlx.Sweep()
			mlx.ClearCache()
		})
	}()

	const (
		bl  = 32
		e   = 128
		top = 4
	)
	if err := thread.Do(context.Background(), func() error {
		data := make([]float32, bl*e)
		for i := range data {
			data[i] = float32(i%17) - 8
		}
		logits := mlx.FromValues(data, bl, e)
		inds := mlx.Argpartition(mlx.Neg(logits), top-1, -1)
		shape := inds.Dims()
		if len(shape) != 2 || shape[0] != bl || shape[1] != e {
			t.Fatalf("2D argpartition shape = %v, want [%d %d]", shape, bl, e)
		}
		inds = mlx.SliceStartStop(inds, []int32{0, 0}, []int32{bl, top})
		got := inds.Dims()
		if len(got) != 2 || got[0] != bl || got[1] != top {
			t.Fatalf("sliced shape = %v, want [%d %d]", got, bl, top)
		}

		logits3 := mlx.Reshape(logits, 1, bl, e)
		inds3 := mlx.Argpartition(mlx.Neg(logits3), top-1, -1)
		t.Logf("rank-3 argpartition dims=%v", inds3.Dims())
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
