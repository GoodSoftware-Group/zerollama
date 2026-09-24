package vae

import (
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

func TestExportNCHWFromNHWC(t *testing.T) {
	if err := mlx.InitMLX(); err != nil {
		t.Skipf("mlx unavailable: %v", err)
	}
	// NHWC [1,2,2,3]: pixel (y,x) = (10*y+x+1, 20*y+x+1, 30*y+x+1)/255 conceptually as floats.
	nhwc := []float32{
		0.10, 0.20, 0.30, // (0,0)
		0.11, 0.21, 0.31, // (0,1)
		0.12, 0.22, 0.32, // (1,0)
		0.13, 0.23, 0.33, // (1,1)
	}
	in := mlx.NewArray(nhwc, []int32{1, 2, 2, 3})
	mlx.Untrack(in)
	out := ExportNCHWFromNHWC(in)
	defer out.Free()

	shape := out.Shape()
	if len(shape) != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != 2 || shape[3] != 2 {
		t.Fatalf("shape=%v, want [1 3 2 2]", shape)
	}
	got := mlx.HostFloat32Slice(out)
	// NCHW: R plane, G plane, B plane
	want := []float32{
		0.10, 0.11, 0.12, 0.13, // R
		0.20, 0.21, 0.22, 0.23, // G
		0.30, 0.31, 0.32, 0.33, // B
	}
	if len(got) < len(want) {
		t.Fatalf("got %d floats, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("at %d: got %v want %v (full=%v)", i, got[i], w, got[:len(want)])
		}
	}
}
