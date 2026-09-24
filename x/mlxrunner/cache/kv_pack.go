package cache

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// Paged-out trie snapshots (not the live decode cache) pack to FP8 when the
// owned window is large enough. Live affine KV (ZEROLLAMA_MLX_KV_QUANT=4|8)
// stores packed triples in the cache; copyOut denseViews first so this path
// still sees activations. Tiny unit-test snapshots stay unpacked so exact
// Floats() checks keep working.
const pagedKVPackMinBytes = 64 << 10

func packOwnedKV(k, v *mlx.Array) (ok, ov *mlx.Array, packed bool, elem mlx.DType) {
	elem = k.DType()
	if k == nil || v == nil || k.NumBytes()+v.NumBytes() < pagedKVPackMinBytes {
		return k, v, false, elem
	}
	switch elem {
	case mlx.DTypeFloat16, mlx.DTypeBFloat16, mlx.DTypeFloat32:
	default:
		return k, v, false, elem
	}
	pk, pv := mlx.ToFP8(k), mlx.ToFP8(v)
	mlx.Eval(pk, pv)
	if pk.NumBytes()+pv.NumBytes() >= k.NumBytes()+v.NumBytes() {
		return k, v, false, elem
	}
	mlx.Unpin(k, v)
	mlx.Pin(pk, pv)
	return pk, pv, true, elem
}

func unpackOwnedKV(k, v *mlx.Array, packed bool, elem mlx.DType) (*mlx.Array, *mlx.Array) {
	if !packed || k == nil || v == nil {
		return k, v
	}
	dk, dv := mlx.FromFP8(k, elem), mlx.FromFP8(v, elem)
	mlx.Eval(dk, dv)
	return dk, dv
}
