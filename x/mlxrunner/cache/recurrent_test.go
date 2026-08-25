package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// TestRecurrentCacheRestoreExactOffset verifies that RecurrentCache restore
// only succeeds when target exactly matches the snapshot's offset. Recurrent
// state is cumulative, so it can't be rewound or fast-forwarded.
func TestRecurrentCacheRestoreExactOffset(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	b1 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1)}
	c.Get(b1, mlx.DTypeFloat16) // lazy-init

	keep := func() ([]*mlx.Array, []*mlx.Array) {
		s := c.State()
		return []*mlx.Array{s[0]}, []*mlx.Array{s[1]}
	}

	b10 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 10), SeqQueryLens: []int32{10}}
	cs, ds := keep()
	c.Put(b10, cs, ds) // advance to 10

	snap := c.Snapshot(0) // snap.offset == 10

	b5 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 5), SeqQueryLens: []int32{5}}
	cs, ds = keep()
	c.Put(b5, cs, ds) // cache now at 15

	// target < snap.offset: fails (can't rewind past snapshot)
	if c.Restore(snap, 5) {
		t.Fatal("Restore(snap, 5) should fail — target != snap.offset")
	}

	// target > snap.offset: fails (can't advance without feeding tokens)
	if c.Restore(snap, 15) {
		t.Fatal("Restore(snap, 15) should fail — target != snap.offset")
	}

	// target == snap.offset: succeeds
	if !c.Restore(snap, 10) {
		t.Fatal("Restore(snap, 10) should succeed — target == snap.offset")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c.Offset())
	}
}

func TestRecurrentCacheGetLazyInit(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 4, 2, 4, 4)
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 1),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
	}
	h := c.Get(b, mlx.DTypeBFloat16)
	if c.Offset() != 0 {
		t.Fatalf("Get should not advance; got offset %d", c.Offset())
	}
	if h.ConvState() == nil || h.DeltaState() == nil {
		t.Fatal("history should expose conv/delta tensors")
	}
	if got := h.ConvState().DType(); got != mlx.DTypeBFloat16 {
		t.Fatalf("conv state dtype = %v, want %v", got, mlx.DTypeBFloat16)
	}
	if got := h.DeltaState().DType(); got != mlx.DTypeFloat32 {
		t.Fatalf("delta state dtype = %v, want %v", got, mlx.DTypeFloat32)
	}
}

func TestRecurrentCachePutAdvances(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 4, 2, 4, 4)
	b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 2), SeqQueryLens: []int32{2}}
	newConv := mlx.Zeros(mlx.DTypeFloat16, 1, 3, 4)
	newDelta := mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4, 4)
	c.Put(b, []*mlx.Array{newConv}, []*mlx.Array{newDelta})
	if c.Offset() != 2 {
		t.Fatalf("cache offset not advanced: %d", c.Offset())
	}
}
