package mlx

import "testing"

func TestGatherQMMAffineSmoke(t *testing.T) {
	if err := CheckInit(); err != nil {
		t.Skipf("mlx unavailable: %v", err)
	}
	path, _ := LoadedLibraryPath()
	t.Logf("lib=%s", path)

	withMLXThread(t, func() {
		// Affine bits=8 group=64 shaped like gemma4 optiq gather_qmm.
		x := Zeros(DTypeFloat16, 4, 1, 1, 64)
		w := Zeros(DTypeUint32, 2, 64, 16)
		scales := Zeros(DTypeFloat16, 2, 64, 1)
		biases := Zeros(DTypeFloat16, 2, 64, 1)
		idx := NewArrayInt32([]int32{0, 1, 0, 1}, []int32{4, 1})
		out := GatherQMM(x, w, scales, biases, nil, idx, true, 64, 8, "affine", true)
		Eval(out)
		dims := out.Dims()
		if len(dims) == 0 {
			t.Fatal("empty output")
		}
		t.Logf("ok dims=%v", dims)
	})
}
