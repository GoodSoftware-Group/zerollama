package siglip

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestTextAndVisionForwardShapes(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	mlx.Scoped(func() {
		const hidden, heads, inner int32 = 8, 2, 16
		block := tinyBlock(hidden, inner)
		text := &textTower{
			Tokens:    nn.NewEmbedding(mlx.FromValues(fill(16*int(hidden), 0.01), 16, int(hidden))),
			Positions: nn.NewEmbedding(mlx.FromValues(fill(4*int(hidden), 0.02), 4, int(hidden))),
			Blocks:    []*encoderBlock{block},
			FinalNorm: &nn.LayerNorm{Weight: mlx.FromValues(fill(int(hidden), 1), int(hidden)), Bias: mlx.FromValues(fill(int(hidden), 0), int(hidden)), Eps: 1e-6},
			Head:      nn.NewLinear(mlx.FromValues(eye(int(hidden)), int(hidden), int(hidden)), nil),
			cfg:       &TextConfig{HiddenSize: hidden, NumHeads: heads, MaxPositionEmbeddings: 4, ProjectionSize: hidden, HiddenAct: "gelu_pytorch_tanh"},
			act:       activation("gelu_pytorch_tanh"),
		}
		out := text.forward(mlx.FromValues([]int32{3, 4}, 1, 2), 1)
		mlx.Eval(out)
		if out.NumDims() != 3 || out.Dim(0) != 1 || out.Dim(1) != 1 || out.Dim(2) != int(hidden) {
			t.Fatalf("text shape %v", out.Dims())
		}

		visionBlock := tinyBlock(hidden, inner)
		weight := mlx.FromValues(fill(int(hidden)*2*2*3, 0.01), int(hidden), 2, 2, 3)
		vision := &visionTower{
			PatchWeight: weight,
			Positions:   nn.NewEmbedding(mlx.FromValues(fill(4*int(hidden), 0.02), 4, int(hidden))),
			Blocks:      []*encoderBlock{visionBlock},
			PostNorm:    &nn.LayerNorm{Weight: mlx.FromValues(fill(int(hidden), 1), int(hidden)), Bias: mlx.FromValues(fill(int(hidden), 0), int(hidden)), Eps: 1e-6},
			Head: &mapHead{
				Probe:    mlx.FromValues(fill(int(hidden), 0.1), 1, 1, int(hidden)),
				InWeight: mlx.FromValues(fill(3*int(hidden)*int(hidden), 0.01), 3*int(hidden), int(hidden)),
				Out:      nn.NewLinear(mlx.FromValues(eye(int(hidden)), int(hidden), int(hidden)), nil),
				Norm:     &nn.LayerNorm{Weight: mlx.FromValues(fill(int(hidden), 1), int(hidden)), Bias: mlx.FromValues(fill(int(hidden), 0), int(hidden)), Eps: 1e-6},
				FC1:      nn.NewLinear(mlx.FromValues(fill(int(inner)*int(hidden), 0.01), int(inner), int(hidden)), nil),
				FC2:      nn.NewLinear(mlx.FromValues(fill(int(hidden)*int(inner), 0.01), int(hidden), int(inner)), nil),
				heads:    heads,
				act:      activation("gelu_pytorch_tanh"),
			},
			cfg: &VisionConfig{HiddenSize: hidden, NumHeads: heads, NumChannels: 3, PatchSize: 2, NumPatches: 4, HiddenAct: "gelu_pytorch_tanh"},
			act: activation("gelu_pytorch_tanh"),
		}
		pixels := fill(1*4*4*3, 0.2)
		got := vision.encode(imagePlan{Pixels: pixels, Height: 4, Width: 4, GridH: 2, GridW: 2, PatchSize: 2})
		mlx.Eval(got)
		if got.NumDims() != 3 || got.Dim(0) != 1 || got.Dim(1) != 1 || got.Dim(2) != int(hidden) {
			t.Fatalf("vision shape %v", got.Dims())
		}
	})
}

func tinyBlock(hidden, inner int32) *encoderBlock {
	return &encoderBlock{
		Norm1: &nn.LayerNorm{Weight: mlx.FromValues(fill(int(hidden), 1), int(hidden)), Bias: mlx.FromValues(fill(int(hidden), 0), int(hidden)), Eps: 1e-6},
		Norm2: &nn.LayerNorm{Weight: mlx.FromValues(fill(int(hidden), 1), int(hidden)), Bias: mlx.FromValues(fill(int(hidden), 0), int(hidden)), Eps: 1e-6},
		Q:     nn.NewLinear(mlx.FromValues(eye(int(hidden)), int(hidden), int(hidden)), nil),
		K:     nn.NewLinear(mlx.FromValues(eye(int(hidden)), int(hidden), int(hidden)), nil),
		V:     nn.NewLinear(mlx.FromValues(eye(int(hidden)), int(hidden), int(hidden)), nil),
		Out:   nn.NewLinear(mlx.FromValues(eye(int(hidden)), int(hidden), int(hidden)), nil),
		FC1:   nn.NewLinear(mlx.FromValues(fill(int(inner)*int(hidden), 0.01), int(inner), int(hidden)), nil),
		FC2:   nn.NewLinear(mlx.FromValues(fill(int(hidden)*int(inner), 0.01), int(hidden), int(inner)), nil),
	}
}

func fill(n int, v float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func eye(n int) []float32 {
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		out[i*n+i] = 1
	}
	return out
}
