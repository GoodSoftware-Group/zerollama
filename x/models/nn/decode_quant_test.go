package nn

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestIsDecodeActivation(t *testing.T) {
	skipIfNoMLX(t)
	prefill := mlx.Zeros(mlx.DTypeFloat32, 1, 8, 64)
	decode := mlx.Zeros(mlx.DTypeFloat32, 1, 1, 64)
	spec := mlx.Zeros(mlx.DTypeFloat32, 1, 6, 64)
	if isDecodeActivation(prefill) || isDecodeActivation(spec) {
		t.Fatal("prefill/spec fused must stay dense")
	}
	if !isDecodeActivation(decode) {
		t.Fatal("L=1 must use decode path")
	}
}

func TestDecodeQuantPrefillMatchesDense(t *testing.T) {
	skipIfNoMLX(t)
	const out, in = 64, 64
	w := mlx.Zeros(mlx.DTypeFloat32, out, in)
	x := mlx.Zeros(mlx.DTypeFloat32, 1, 8, in)
	mlx.Eval(w, x)
	dense := NewLinear(w, nil)
	wrapped := WrapDecodeQuant(dense).(*DecodeQuantLinear)
	got := wrapped.Forward(x)
	want := dense.Forward(x)
	mlx.Eval(got, want)
	gs, ws := got.Floats(), want.Floats()
	if len(gs) != len(ws) {
		t.Fatalf("len %d vs %d", len(gs), len(ws))
	}
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("prefill must be bit-identical to dense at %d: %v vs %v", i, gs[i], ws[i])
		}
	}
}

func TestQuantizeLinearLayerDropsDenseCopy(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.Zeros(mlx.DTypeFloat32, 64, 64)
	mlx.Eval(w)
	wrapped := WrapDecodeQuant(NewLinear(w, nil)).(*DecodeQuantLinear)
	got := QuantizeLinearLayer(wrapped)
	if _, ok := got.(*QuantizedLinear); !ok {
		t.Fatalf("want QuantizedLinear, got %T", got)
	}
}

func TestWrapDecodeQuantSkipsBadInDim(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.Zeros(mlx.DTypeFloat32, 32, 48) // 48 % 64 != 0
	mlx.Eval(w)
	got := WrapDecodeQuant(NewLinear(w, nil))
	if _, ok := got.(*Linear); !ok {
		t.Fatalf("want dense Linear, got %T", got)
	}
}
