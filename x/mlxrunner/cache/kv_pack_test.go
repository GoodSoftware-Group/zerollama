package cache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPagedKVPackRoundTrip(t *testing.T) {
	skipIfNoMLX(t)
	// Cross the pack threshold: 1×8×512×32 float32 ×2 ≈ 1 MiB.
	const H, L, D = 8, 512, 32
	n := H * L * D
	ks := make([]float32, n)
	vs := make([]float32, n)
	for i := range ks {
		ks[i] = 0.5
		vs[i] = -0.25
	}
	k := mlx.FromValues(ks, 1, H, L, D)
	v := mlx.FromValues(vs, 1, H, L, D)
	mlx.Eval(k, v)
	mlx.Pin(k, v)
	dense := k.NumBytes() + v.NumBytes()

	pk, pv, packed, elem := packOwnedKV(k, v)
	if !packed {
		t.Fatal("expected FP8 pack on large KV snapshot")
	}
	if pk.NumBytes()+pv.NumBytes() >= dense {
		t.Fatalf("packed %d bytes, dense %d", pk.NumBytes()+pv.NumBytes(), dense)
	}

	dk, dv := unpackOwnedKV(pk, pv, packed, elem)
	mlx.Eval(dk, dv)
	got := dk.Floats()
	for i := range 32 {
		if math.Abs(float64(got[i]-0.5)) > 0.08 {
			t.Fatalf("unpacked key[%d]=%v, want ~0.5", i, got[i])
		}
	}
	gv := dv.Floats()
	if math.Abs(float64(gv[0]+0.25)) > 0.08 {
		t.Fatalf("unpacked value[0]=%v, want ~-0.25", gv[0])
	}
}
