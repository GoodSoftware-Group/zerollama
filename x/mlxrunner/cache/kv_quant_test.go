package cache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestKVQuantFromEnv(t *testing.T) {
	t.Setenv("ZEROLLAMA_MLX_KV_QUANT", "")
	if c := KVQuantFromEnv(); c.IsAffine() {
		t.Fatal("empty env should be off")
	}
	t.Setenv("ZEROLLAMA_MLX_KV_QUANT", "8")
	if c := KVQuantFromEnv(); !c.IsAffine() || c.Bits != 8 || c.GroupSize != 64 {
		t.Fatalf("got %+v", c)
	}
	t.Setenv("ZEROLLAMA_MLX_KV_QUANT", "4")
	if c := KVQuantFromEnv(); c.Bits != 4 {
		t.Fatalf("got bits %d", c.Bits)
	}
}

func TestAffineQuantRoundTrip(t *testing.T) {
	skipIfNoMLX(t)
	cfg := AffineKVQuant(8)
	const B, H, L, D = 1, 2, 4, 64
	n := B * H * L * D
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i%17) * 0.1
	}
	dense := mlx.FromValues(data, B, H, L, D).AsType(mlx.DTypeBFloat16)
	mlx.Eval(dense)
	q := quantizeAffine(dense, cfg)
	got := dequantizeAffine(q.Q, q.Scales, q.Biases, cfg, mlx.DTypeBFloat16)
	gotF := got.AsType(mlx.DTypeFloat32)
	denseF := dense.AsType(mlx.DTypeFloat32)
	mlx.Eval(gotF, denseF)
	gf := gotF.Floats()
	df := denseF.Floats()
	var maxErr float64
	for i := range gf {
		e := math.Abs(float64(gf[i] - df[i]))
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 0.15 {
		t.Fatalf("affine8 round-trip max err %v", maxErr)
	}
}

func TestKVCacheAffineUpdateView(t *testing.T) {
	skipIfNoMLX(t)
	cfg := AffineKVQuant(8)
	c := NewKVCacheWithQuant(cfg)
	defer c.Free()

	const B, H, L, D = 1, 2, 3, 64
	makeKV := func(base float32) (*mlx.Array, *mlx.Array) {
		n := B * H * L * D
		ks := make([]float32, n)
		vs := make([]float32, n)
		for i := range ks {
			ks[i] = base + float32(i%7)*0.01
			vs[i] = base*0.5 + float32(i%5)*0.02
		}
		k := mlx.FromValues(ks, B, H, L, D).AsType(mlx.DTypeBFloat16)
		v := mlx.FromValues(vs, B, H, L, D).AsType(mlx.DTypeBFloat16)
		mlx.Eval(k, v)
		return k, v
	}

	k1, v1 := makeKV(1)
	hist := c.Update(nil, k1, v1)
	if hist == nil || c.Offset() != L {
		t.Fatalf("offset=%d hist=%v", c.Offset(), hist)
	}
	if !c.affine.initialized() {
		t.Fatal("expected affine live store")
	}
	if c.keys != nil {
		t.Fatal("dense keys should stay nil under affine")
	}

	state := c.State()
	mlx.Eval(state[0], state[1])
	if state[0].Dim(2) != L || state[0].DType() != mlx.DTypeBFloat16 {
		t.Fatalf("denseView shape/dtype %+v %v", state[0].Dims(), state[0].DType())
	}

	k2, v2 := makeKV(2)
	c.Update(nil, k2, v2)
	if c.Offset() != 2*L {
		t.Fatalf("offset=%d", c.Offset())
	}
	view := c.View((*batch.Batch)(nil))
	if view == nil {
		t.Fatal("nil view")
	}

	// Snapshot + restore round-trip through dense owned copy.
	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("nil snapshot")
	}
	snap.(*kvSnapshot).copyOut()
	if !c.Restore(snap, 2*L) {
		t.Fatal("restore failed")
	}
	if c.Offset() != 2*L {
		t.Fatalf("after restore offset=%d", c.Offset())
	}
}

func TestKVCacheAffineHeadDimFallback(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCacheWithQuant(AffineKVQuant(8))
	defer c.Free()
	// head_dim 80 is not divisible by 64 → fall back to dense.
	k := mlx.Zeros(mlx.DTypeBFloat16, 1, 1, 2, 80)
	v := mlx.Zeros(mlx.DTypeBFloat16, 1, 1, 2, 80)
	mlx.Eval(k, v)
	c.Update(nil, k, v)
	if c.quant.IsAffine() {
		t.Fatal("expected fallback to dense")
	}
	if c.keys == nil {
		t.Fatal("expected dense keys after fallback")
	}
}

func TestRotatingKVCacheAffineUpdate(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCacheWithQuant(8, AffineKVQuant(8))
	defer c.Free()

	const B, H, D = 1, 1, 64
	for i := 0; i < 10; i++ {
		k := mlx.Zeros(mlx.DTypeBFloat16, B, H, 1, D)
		v := mlx.Zeros(mlx.DTypeBFloat16, B, H, 1, D)
		mlx.Eval(k, v)
		c.Update(nil, k, v)
	}
	if c.Offset() != 10 {
		t.Fatalf("offset=%d", c.Offset())
	}
	if !c.affine.initialized() {
		t.Fatal("expected affine store")
	}
	st := c.State()
	mlx.Eval(st[0], st[1])
	if st[0].Dim(2) != 8 { // maxSize
		t.Fatalf("window len %d", st[0].Dim(2))
	}
}
