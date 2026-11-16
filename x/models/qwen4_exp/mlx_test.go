package qwen4_exp

import "testing"

import "github.com/ollama/ollama/x/mlxrunner/mlx"

func runMLX(t *testing.T, fn func(t *testing.T)) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	fn(t)
}
