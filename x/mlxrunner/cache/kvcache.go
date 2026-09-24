package cache

import (
	"slices"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// Attention is the contract for caches that back attention layers
// (KVCache, RotatingKVCache).
type Attention interface {
	Cache

	// Update appends (k, v) and returns an opaque nn.KVHistory for
	// this layer's SDPA.
	Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory

	// View returns the current attention history without writing.
	View(b *batch.Batch) *nn.KVHistory
}

type KVCache struct {
	keys, values *mlx.Array
	affine       affineBuffers
	quant        KVQuantConfig
	denseElem    mlx.DType
	offset       int
	step         int

	snapshots pendingSnapshots

	// lazySnapshots index into the live keys/values buffer rather than owning a
	// copy (see kvSnapshot); the cache copies them out before overwriting or
	// freeing the slots they name.
	lazySnapshots []*kvSnapshot
}

func NewKVCache() *KVCache {
	return &KVCache{step: 256, quant: KVQuantFromEnv()}
}

// NewKVCacheWithQuant is for tests that pin a scheme without env.
func NewKVCacheWithQuant(cfg KVQuantConfig) *KVCache {
	return &KVCache{step: 256, quant: cfg}
}

// Assumes B = 1; heterogeneous batches are not supported.
func (c *KVCache) Update(_ *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	start := c.offset
	newK, newV := c.appendKV(keys, values)
	c.captureLazySnapshots(start, c.offset)
	return nn.NewKVHistory(newK, newV, nil)
}

// appendKV is the raw write path shared by Update and Restore.
func (c *KVCache) appendKV(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	if c.keys == nil && c.affine.kq == nil && c.quant.IsAffine() {
		if !headDimOK(keys.Dim(3), c.quant.GroupSize) || !headDimOK(values.Dim(3), c.quant.GroupSize) {
			logutil.Warn("mlx KV quant disabled: head_dim not divisible by group size",
				"k_dim", keys.Dim(3), "v_dim", values.Dim(3), "group", c.quant.GroupSize)
			c.quant = DenseKVQuantConfig()
		}
	}
	if c.quant.IsAffine() {
		return c.appendKVAffine(keys, values)
	}
	return c.appendKVDense(keys, values)
}

func (c *KVCache) appendKVDense(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	B, H, L, Dk, Dv := keys.Dim(0), keys.Dim(1), keys.Dim(2), keys.Dim(3), values.Dim(3)

	prev := c.offset

	for _, s := range slices.Clone(c.lazySnapshots) {
		if s.fromOffset < prev+L && s.toOffset > prev {
			s.copyOut()
		}
	}

	if c.keys == nil || (prev+L) > c.keys.Dim(2) {
		steps := (c.step + L - 1) / c.step
		newKeys := mlx.Zeros(keys.DType(), B, H, steps*c.step, Dk)
		newValues := mlx.Zeros(values.DType(), B, H, steps*c.step, Dv)

		if c.keys != nil {
			if prev%c.step != 0 {
				c.keys.Set(c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, prev), mlx.Slice()))
				c.values.Set(c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, prev), mlx.Slice()))
			}
			c.keys.Set(c.keys.Concatenate(2, newKeys))
			c.values.Set(c.values.Concatenate(2, newValues))
		} else {
			c.keys, c.values = newKeys, newValues
			c.denseElem = keys.DType()
			mlx.Pin(c.keys, c.values)
		}
	}

	c.offset += L
	c.keys.Set(c.keys.SliceUpdate(keys, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, c.offset), mlx.Slice()))
	c.values.Set(c.values.SliceUpdate(values, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, c.offset), mlx.Slice()))

	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice())
}

func (c *KVCache) appendKVAffine(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	B, H, L := keys.Dim(0), keys.Dim(1), keys.Dim(2)
	prev := c.offset

	for _, s := range slices.Clone(c.lazySnapshots) {
		if s.fromOffset < prev+L && s.toOffset > prev {
			s.copyOut()
		}
	}

	kq := quantizeAffine(keys, c.quant)
	vq := quantizeAffine(values, c.quant)

	if !c.affine.initialized() || (prev+L) > c.affine.seqCap() {
		steps := (c.step + L - 1) / c.step
		grown := growAffineZeros(B, H, steps*c.step,
			kq.Q.Dim(3), vq.Q.Dim(3), kq.Scales.Dim(3), vq.Scales.Dim(3))

		if c.affine.initialized() {
			if prev%c.step != 0 {
				c.affine.setSliceSeq(c.affine.sliceSeq(0, prev))
			}
			c.affine.concatenate(2, grown)
		} else {
			c.affine = grown
			c.denseElem = keys.DType()
			c.affine.pin()
		}
	}

	c.offset += L
	c.affine.writeAt(prev, c.offset, kq, vq)

	return c.affine.denseSlice(c.quant, c.denseElem, 0, c.offset)
}

// View returns the current cache contents as attention history without writing.
func (c *KVCache) View(_ *batch.Batch) *nn.KVHistory {
	state := c.State()
	if state == nil {
		return nn.NewKVHistory(nil, nil, nil)
	}
	return nn.NewKVHistory(state[0], state[1], nil)
}

func (c *KVCache) State() []*mlx.Array {
	if c.quant.IsAffine() {
		if !c.affine.initialized() || c.offset == 0 {
			return nil
		}
		k, v := c.affine.denseSlice(c.quant, c.denseElem, 0, c.offset)
		return []*mlx.Array{k, v}
	}
	if c.keys == nil || c.values == nil {
		return nil
	}
	return []*mlx.Array{
		c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, c.offset), mlx.Slice()),
	}
}

func (c *KVCache) PrepareSnapshots(offsets []int) { c.snapshots.prepare(c.offset, offsets) }
func (c *KVCache) TakeSnapshots() []Snapshot      { return c.snapshots.take() }

func (c *KVCache) captureLazySnapshots(start, end int) {
	for _, o := range c.snapshots.scheduledIn(start, end) {
		c.snapshots.captureReached(o, func(int) Snapshot { return c.lazySnapshot(c.snapshots.base, o) })
	}
}

type kvSnapshot struct {
	keys, values         *mlx.Array
	fromOffset, toOffset int
	cache                *KVCache

	packed bool
	elem   mlx.DType

	onMaterialize func(delta int)
}

func (s *kvSnapshot) Size() int {
	if s.keys != nil {
		return s.keys.NumBytes() + s.values.NumBytes()
	}
	return 0
}

func (s *kvSnapshot) SetMaterializeHook(fn func(delta int)) { s.onMaterialize = fn }

func (s *kvSnapshot) Close() {
	mlx.Unpin(s.keys, s.values)
	if s.cache != nil {
		s.cache.dropLazySnapshot(s)
		s.cache = nil
	}
}

func (s *kvSnapshot) copyOut() {
	if s.keys != nil {
		return
	}
	c := s.cache
	var kCopy, vCopy *mlx.Array
	if c.quant.IsAffine() {
		dk, dv := c.affine.denseSlice(c.quant, c.denseElem, s.fromOffset, s.toOffset)
		kCopy = mlx.Contiguous(dk, false)
		vCopy = mlx.Contiguous(dv, false)
	} else {
		kSlice := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.fromOffset, s.toOffset), mlx.Slice())
		vSlice := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.fromOffset, s.toOffset), mlx.Slice())
		kCopy = mlx.Contiguous(kSlice, false)
		vCopy = mlx.Contiguous(vSlice, false)
	}
	mlx.Pin(kCopy, vCopy)
	mlx.AsyncEval(kCopy, vCopy)
	kCopy, vCopy, s.packed, s.elem = packOwnedKV(kCopy, vCopy)

	s.keys, s.values = kCopy, vCopy
	c.dropLazySnapshot(s)
	s.cache = nil

	if s.onMaterialize != nil {
		s.onMaterialize(s.keys.NumBytes() + s.values.NumBytes())
		s.onMaterialize = nil
	}
}

func (c *KVCache) addLazySnapshot(s *kvSnapshot) { c.lazySnapshots = append(c.lazySnapshots, s) }

func (c *KVCache) dropLazySnapshot(s *kvSnapshot) {
	if i := slices.Index(c.lazySnapshots, s); i >= 0 {
		c.lazySnapshots = slices.Delete(c.lazySnapshots, i, i+1)
	}
}

func (c *KVCache) Snapshot(fromOffset int) Snapshot {
	return c.lazySnapshot(fromOffset, c.offset)
}

func (c *KVCache) lazySnapshot(fromOffset, toOffset int) Snapshot {
	live := c.keys != nil || c.affine.initialized()
	if !live || toOffset <= fromOffset {
		return nil
	}
	s := &kvSnapshot{
		fromOffset: fromOffset,
		toOffset:   toOffset,
		cache:      c,
	}
	c.addLazySnapshot(s)
	return s
}

func (c *KVCache) Restore(snapshot Snapshot, target int) bool {
	if target < 0 {
		return false
	}

	if snapshot == nil {
		if target > c.offset {
			return false
		}
		c.offset = target
		return true
	}

	snap := snapshot.(*kvSnapshot)

	if target > snap.toOffset || c.offset < snap.fromOffset {
		return false
	}

	if snap.cache == c && snap.keys == nil {
		c.offset = min(target, snap.toOffset)
		return true
	}

	snap.copyOut()

	c.offset = snap.fromOffset
	rk, rv := unpackOwnedKV(snap.keys, snap.values, snap.packed, snap.elem)
	c.appendKV(rk, rv)

	if target < c.offset {
		c.offset = target
	}

	return true
}

func (c *KVCache) Merge(parent, child Snapshot) Snapshot {
	if parent == nil || child == nil {
		if parent != nil {
			parent.Close()
		}
		if child != nil {
			child.Close()
		}
		return nil
	}
	p := parent.(*kvSnapshot)
	ch := child.(*kvSnapshot)

	if p.keys == nil && ch.keys == nil && p.cache == ch.cache && p.toOffset == ch.fromOffset {
		merged := &kvSnapshot{fromOffset: p.fromOffset, toOffset: ch.toOffset, cache: p.cache}
		p.cache.addLazySnapshot(merged)
		p.Close()
		ch.Close()
		return merged
	}

	p.copyOut()
	ch.copyOut()

	pk, pv := unpackOwnedKV(p.keys, p.values, p.packed, p.elem)
	ck, cv := unpackOwnedKV(ch.keys, ch.values, ch.packed, ch.elem)
	mk := pk.Concatenate(2, ck)
	mv := pv.Concatenate(2, cv)
	mlx.Pin(mk, mv)
	mlx.AsyncEval(mk, mv)
	mk, mv, packed, elem := packOwnedKV(mk, mv)

	p.Close()
	ch.Close()

	return &kvSnapshot{
		keys:       mk,
		values:     mv,
		fromOffset: p.fromOffset,
		toOffset:   ch.toOffset,
		packed:     packed,
		elem:       elem,
	}
}

func (c *KVCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	if snapshot == nil {
		return nil, nil
	}
	snap := snapshot.(*kvSnapshot)
	splitIdx := at - snap.fromOffset
	seqLen := snap.toOffset - snap.fromOffset
	if splitIdx <= 0 {
		return nil, snapshot
	}
	if splitIdx >= seqLen {
		return snapshot, nil
	}

	if snap.keys == nil {
		p := &kvSnapshot{fromOffset: snap.fromOffset, toOffset: at, cache: snap.cache}
		ch := &kvSnapshot{fromOffset: at, toOffset: snap.toOffset, cache: snap.cache}
		snap.cache.addLazySnapshot(p)
		snap.cache.addLazySnapshot(ch)
		snap.Close()
		return p, ch
	}

	dk, dv := unpackOwnedKV(snap.keys, snap.values, snap.packed, snap.elem)
	pk := mlx.Contiguous(dk.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, splitIdx), mlx.Slice()), false)
	pv := mlx.Contiguous(dv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, splitIdx), mlx.Slice()), false)
	ck := mlx.Contiguous(dk.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(splitIdx, seqLen), mlx.Slice()), false)
	cv := mlx.Contiguous(dv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(splitIdx, seqLen), mlx.Slice()), false)
	mlx.Pin(pk, pv, ck, cv)
	mlx.AsyncEval(pk, pv, ck, cv)
	pk, pv, pPacked, pElem := packOwnedKV(pk, pv)
	ck, cv, cPacked, cElem := packOwnedKV(ck, cv)

	snap.Close()

	p := &kvSnapshot{
		keys:       pk,
		values:     pv,
		fromOffset: snap.fromOffset,
		toOffset:   at,
		packed:     pPacked,
		elem:       pElem,
	}
	ch := &kvSnapshot{
		keys:       ck,
		values:     cv,
		fromOffset: at,
		toOffset:   snap.toOffset,
		packed:     cPacked,
		elem:       cElem,
	}
	return p, ch
}

func (c *KVCache) Free() {
	for _, s := range slices.Clone(c.lazySnapshots) {
		s.copyOut()
	}
	if c.quant.IsAffine() {
		c.affine.unpin()
		c.affine.clear()
	} else {
		mlx.Unpin(c.keys, c.values)
		c.keys, c.values = nil, nil
	}
	c.offset = 0
	c.snapshots = pendingSnapshots{}
}

func (c *KVCache) Offset() int { return c.offset }
