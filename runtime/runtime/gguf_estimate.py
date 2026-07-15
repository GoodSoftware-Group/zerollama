"""Estimate GGUF weight bytes from file metadata (Phase 13)."""

from __future__ import annotations

import os
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# Unknown ggml KV types: assume fp16 (conservative vs 1-byte quant guesses).
_GGML_KV_UNKNOWN_BYTES = 2

_hints_cache: dict[str, tuple[float, float, GgufArchHints]] = {}
_hints_lock = threading.Lock()
_HINTS_CACHE_TTL_S = 60.0


@dataclass
class GgufArchHints:
    """Architecture fields read from GGUF metadata (scalar + optional per-layer arrays)."""

    scalar: dict[str, int] = field(default_factory=dict)
    sliding_window_per_layer: tuple[int, ...] | None = None
    head_count_kv_per_layer: tuple[int, ...] | None = None
    head_dim_source: str | None = None

    def get(self, key: str, default: int | None = None) -> int | None:
        return self.scalar.get(key, default)

    def __getitem__(self, key: str) -> int:
        return self.scalar[key]


def _read_exact(f, n: int) -> bytes:
    """Read exactly *n* bytes or raise (corrupt/truncated GGUF)."""
    data = f.read(n)
    if not isinstance(data, (bytes, bytearray)) or len(data) != n:
        raise EOFError(f"gguf short read: wanted {n} bytes")
    return bytes(data)


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f) -> str:
    n = _read_u64(f)
    return f.read(n).decode("utf-8", errors="replace")


def _skip_typed_value(f, type_id: int) -> None:
    if type_id == GGUF_TYPE_STRING:
        n = _read_u64(f)
        f.seek(n, 1)
        return
    if type_id == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        for _ in range(count):
            _skip_typed_value(f, elem_type)
        return
    if type_id == GGUF_TYPE_UINT32:
        f.seek(4, 1)
        return
    sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    n = sizes.get(type_id)
    if n is None:
        raise ValueError(f"unsupported gguf value type {type_id}")
    f.seek(n, 1)


def _read_gguf_value(f, type_id: int) -> int | list[int] | None:
    if type_id == GGUF_TYPE_STRING:
        n = _read_u64(f)
        f.seek(n, 1)
        return None
    if type_id == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        if elem_type != GGUF_TYPE_UINT32 or count == 0:
            for _ in range(count):
                _skip_typed_value(f, elem_type)
            return None
        return [_read_u32(f) for _ in range(count)]
    if type_id == GGUF_TYPE_UINT32:
        return _read_u32(f)
    _skip_typed_value(f, type_id)
    return None


def _hint_from_gguf_key(key: str, val: int, hints: dict[str, int]) -> None:
    """Map llama.cpp GGUF metadata keys into scalar hints."""
    if key.endswith(".block_count"):
        hints["block_count"] = val
    elif key.endswith(".context_length"):
        hints["context_length"] = val
    elif key.endswith(".embedding_length"):
        hints["embedding_length"] = val
    elif key.endswith(".attention.head_count_kv"):
        hints["head_count_kv"] = val
    elif key.endswith(".attention.head_count"):
        hints["head_count"] = val
    elif key.endswith(".attention.key_type"):
        hints["key_type"] = val
    elif key.endswith(".attention.value_type"):
        hints["value_type"] = val
    elif key.endswith(".attention.key_length"):
        hints["key_length"] = val
    elif key.endswith(".attention.value_length"):
        hints["value_length"] = val
    elif key.endswith(".sliding_window"):
        hints["sliding_window"] = val


def _parse_gguf_arch_hints(path: Path) -> GgufArchHints:
    scalar: dict[str, int] = {}
    sw_layers: tuple[int, ...] | None = None
    kv_layers: tuple[int, ...] | None = None
    try:
        with path.open("rb") as f:
            if _read_u32(f) != GGUF_MAGIC:
                return GgufArchHints()
            version = _read_u32(f)
            if version == 1:
                return GgufArchHints()
            _read_u64(f)  # n_tensors
            n_kv = _read_u64(f)
            for _ in range(n_kv):
                key = _read_string(f)
                vtype = _read_u32(f)
                val = _read_gguf_value(f, vtype)
                if isinstance(val, list):
                    if "sliding_window" in key and len(val) > 0:
                        sw_layers = tuple(int(x) for x in val)
                    elif "head_count_kv" in key and len(val) > 0:
                        kv_layers = tuple(int(x) for x in val)
                    continue
                if isinstance(val, int):
                    _hint_from_gguf_key(key, val, scalar)
    except (OSError, ValueError, struct.error, TypeError, EOFError):
        pass
    return GgufArchHints(
        scalar=scalar,
        sliding_window_per_layer=sw_layers,
        head_count_kv_per_layer=kv_layers,
    )


def gguf_arch_hints(path: Path) -> GgufArchHints:
    """Cached GGUF architecture hints (mtime + TTL)."""
    try:
        resolved = path.resolve()
        mtime = resolved.stat().st_mtime
    except OSError:
        return _parse_gguf_arch_hints(path)

    key = str(resolved)
    now = time.monotonic()
    with _hints_lock:
        cached = _hints_cache.get(key)
        if cached is not None:
            cached_mtime, cached_at, arch = cached
            if cached_mtime == mtime and (now - cached_at) < _HINTS_CACHE_TTL_S:
                return arch

    arch = _parse_gguf_arch_hints(resolved)
    if arch.scalar.get("key_length") is not None or arch.scalar.get("value_length") is not None:
        arch.head_dim_source = "metadata"
    else:
        tensors = gguf_tensor_infos(resolved)
        if tensors:
            inferred = _infer_head_dims_from_tensor_infos(tensors, arch)
            if inferred:
                arch.scalar["key_length"] = inferred[0]
                arch.scalar["value_length"] = inferred[1]
                arch.head_dim_source = "tensor_inferred"
    if arch.head_dim_source is None and gguf_head_dims(arch) is not None:
        arch.head_dim_source = "embedding_fallback"
    with _hints_lock:
        _hints_cache[key] = (mtime, now, arch)
    return arch


def gguf_model_hints(path: Path) -> dict[str, int]:
    """Scalar hints only (backward compatible). Prefer gguf_arch_hints()."""
    return dict(gguf_arch_hints(path).scalar)


def _infer_head_dims_from_tensor_infos(
    infos: list[GgufTensorInfo],
    arch: GgufArchHints,
) -> tuple[int, int] | None:
    """Derive (key_length, value_length) from attn_k / attn_v weight shapes."""
    n_embd = arch.scalar.get("embedding_length") or 0
    n_kv = arch.scalar.get("head_count_kv") or 0
    if n_kv <= 0:
        n_kv = arch.scalar.get("head_count") or 0
    if n_embd <= 0 or n_kv <= 0:
        return None

    def _dim_from_proj(ne: tuple[int, ...]) -> int | None:
        if len(ne) != 2:
            return None
        a, b = ne
        if b == n_embd and a > 0 and a % n_kv == 0:
            return a // n_kv
        if a == n_embd and b > 0 and b % n_kv == 0:
            return b // n_kv
        return None

    def _is_k_tensor(name: str) -> bool:
        nl = name.lower()
        if "attn_k" in nl or nl.endswith("attn_k.weight"):
            return True
        return "k_proj" in nl and ("self_attn" in nl or ".attn." in nl)

    def _is_v_tensor(name: str) -> bool:
        nl = name.lower()
        if "attn_v" in nl or nl.endswith("attn_v.weight"):
            return True
        return "v_proj" in nl and ("self_attn" in nl or ".attn." in nl)

    k_dim: int | None = None
    v_dim: int | None = None
    for t in infos:
        if len(t.ne) != 2:
            continue
        d = _dim_from_proj(t.ne)
        if d is None:
            continue
        if _is_k_tensor(t.name):
            k_dim = d
        elif _is_v_tensor(t.name):
            v_dim = d
    if k_dim and v_dim:
        return k_dim, v_dim
    if k_dim:
        return k_dim, k_dim
    if v_dim:
        return v_dim, v_dim
    return None


def gguf_head_dims(hints: GgufArchHints | dict[str, int]) -> tuple[int, int] | None:
    """Return (key_length, value_length) when known."""
    s = hints.scalar if isinstance(hints, GgufArchHints) else hints
    kd = s.get("key_length")
    vd = s.get("value_length")
    if kd and vd:
        return kd, vd
    if kd and kd > 0:
        return kd, kd
    if vd and vd > 0:
        return vd, vd
    emb = s.get("embedding_length")
    heads = s.get("head_count")
    if emb and heads and heads > 0:
        d = emb // heads
        return d, d
    return None


def gguf_head_dim(hints: GgufArchHints | dict[str, int]) -> int | None:
    dims = gguf_head_dims(hints)
    if dims is None:
        return None
    return max(dims[0], dims[1])


# IQ/TQ/MXFP KV types: use ggml block bytes/element (tighter than flat 2).
_IQ_TQ_KV_TYPE_IDS = frozenset(
    {16, 17, 18, 19, 20, 21, 22, 23, 29, 34, 35, 39}
)
_LEGACY_QUANT_KV_MIN_BYTES = 2

_FLAT_KV_BYTES_TABLE: dict[int, int] = {
    0: 4,  # F32
    1: 2,  # F16
    30: 2,  # BF16
    2: 2,  # Q4_0
    3: 2,  # Q4_1
    6: 2,
    7: 2,
    8: 2,  # Q8_0
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    24: 2,
    25: 2,
    26: 4,
}


def _kv_block_layout_enabled() -> bool:
    from runtime.env import vram_kv_block_layout_enabled

    return vram_kv_block_layout_enabled()


def ggml_type_kv_bytes(type_id: int) -> int:
    """Bytes per KV element; unknown types use fp16 (conservative)."""
    from_layout = _kv_bytes_from_block_layout(type_id)
    if from_layout is not None:
        # llama.cpp often stores KV as F16 even when weight metadata lists IQ/TQ types.
        if type_id in _IQ_TQ_KV_TYPE_IDS:
            return max(from_layout, _GGML_KV_UNKNOWN_BYTES)
        return from_layout
    return _FLAT_KV_BYTES_TABLE.get(type_id, _GGML_KV_UNKNOWN_BYTES)


def kv_bytes_k_and_v(hints: GgufArchHints | dict[str, int], *, default_elem: int = 2) -> tuple[int, int]:
    """Per-element bytes for K and V tensors."""
    s = hints.scalar if isinstance(hints, GgufArchHints) else hints
    kt = s.get("key_type")
    vt = s.get("value_type")
    fallback = max(default_elem, _GGML_KV_UNKNOWN_BYTES)
    if kt is not None:
        k_b = ggml_type_kv_bytes(kt)
        if vt is not None:
            v_b = ggml_type_kv_bytes(vt)
        else:
            v_b = k_b
        return k_b, v_b
    return fallback, fallback


def kv_bytes_per_slot(hints: GgufArchHints | dict[str, int], *, default_elem: int = 2) -> int:
    k_b, v_b = kv_bytes_k_and_v(hints, default_elem=default_elem)
    return k_b + v_b


def kv_cache_type_summary(hints: GgufArchHints | dict[str, int]) -> dict[str, int | bool | str | None]:
    """Per-element KV sizing used by exact KV estimates (for /health)."""
    arch = hints if isinstance(hints, GgufArchHints) else GgufArchHints(scalar=dict(hints))
    k_b, v_b = kv_bytes_k_and_v(arch)
    kt = arch.scalar.get("key_type")
    vt = arch.scalar.get("value_type")
    note: str | None = None
    if kt is not None and kt in _IQ_TQ_KV_TYPE_IDS:
        note = (
            "IQ/TQ/MXFP key_type uses max(block-layout, F16) bytes/element; "
            "autotune/calibration still recommended"
        )
    return {
        "kv_block_layout": _kv_block_layout_enabled(),
        "key_type": kt,
        "value_type": vt,
        "kv_bytes_k": k_b,
        "kv_bytes_v": v_b,
        "kv_bytes_per_slot": k_b + v_b,
        "head_dim": gguf_head_dim(arch),
        "head_dim_source": arch.head_dim_source,
        "note": note,
    }


def _layer_head_count_kv(arch: GgufArchHints, layer_idx: int, default: int) -> int:
    per = arch.head_count_kv_per_layer
    if per and layer_idx < len(per):
        v = per[layer_idx]
        if v > 0:
            return v
    return default


def _layer_context(num_ctx: int, layer_idx: int, arch: GgufArchHints) -> int:
    """Tokens cached at layer_idx (full ctx, scalar SWA, or per-layer array)."""
    per = arch.sliding_window_per_layer
    layers = arch.scalar.get("block_count") or 0
    if per and layers > 0 and len(per) == layers:
        sw = per[layer_idx] if layer_idx < len(per) else 0
        if sw > 0:
            return min(num_ctx, sw)
        return num_ctx
    sw = arch.scalar.get("sliding_window")
    if sw and 0 < sw < num_ctx:
        return sw
    return num_ctx


def estimate_kv_cache_bytes(
    hints: GgufArchHints | dict[str, int],
    num_ctx: int,
    *,
    n_gpu_layers: int | None = None,
    elem_bytes: int | None = None,
) -> int | None:
    """Estimate K+V cache bytes on GPU from GGUF attention metadata."""
    arch = hints if isinstance(hints, GgufArchHints) else GgufArchHints(scalar=dict(hints))
    s = arch.scalar
    layers = s.get("block_count")
    if not layers or layers <= 0 or num_ctx <= 0:
        return None
    default_n_kv = s.get("head_count_kv") or s.get("head_count")
    if (not default_n_kv or default_n_kv <= 0) and arch.head_count_kv_per_layer:
        default_n_kv = max(arch.head_count_kv_per_layer)
    if not default_n_kv or default_n_kv <= 0:
        return None
    dims = gguf_head_dims(arch)
    if not dims:
        return None
    k_dim, v_dim = dims
    if elem_bytes is None:
        from runtime.env import vram_kv_elem_bytes

        elem_bytes = vram_kv_elem_bytes()
    k_b, v_b = kv_bytes_k_and_v(arch, default_elem=elem_bytes)

    total = 0
    for layer in range(layers):
        if n_gpu_layers is not None and n_gpu_layers >= 0 and layer >= n_gpu_layers:
            continue
        n_kv = _layer_head_count_kv(arch, layer, default_n_kv)
        ctx_l = _layer_context(num_ctx, layer, arch)
        total += ctx_l * n_kv * (k_dim * k_b + v_dim * v_b)
    return total


_BLK_LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)\.")

# (block_size, type_size) from ggml-common.h (QK_K=256, QK4_NL=32, QK_MXFP4=32).
_GGML_BLOCK_LAYOUT: dict[int, tuple[int, int]] = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    29: (256, 56),  # IQ1_M
    30: (1, 2),
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
    40: (64, 36),  # NVFP4
    # WHY (32,34): F16 scale + 32 FP8 bytes — same footprint as Q8_0; VRAM estimate must not treat as F16.
    51: (32, 34),  # FP8_E4M3
    52: (32, 34),  # FP8_E5M2
}


def _kv_bytes_from_block_layout(type_id: int) -> int | None:
    """Block-layout KV bytes/element when enabled; None → use flat table."""
    if not _kv_block_layout_enabled():
        return None
    layout = _GGML_BLOCK_LAYOUT.get(type_id)
    if layout is None:
        return None
    blck, type_size = layout
    flat = _FLAT_KV_BYTES_TABLE.get(type_id)
    if blck > 1:
        per = max(1, -(-type_size // blck))
        if type_id in _IQ_TQ_KV_TYPE_IDS:
            return min(per, _GGML_KV_UNKNOWN_BYTES)
        return max(_LEGACY_QUANT_KV_MIN_BYTES, per)
    if blck == 1:
        if type_id in _IQ_TQ_KV_TYPE_IDS:
            return max(1, type_size)
        if flat is not None:
            return max(flat, type_size)
        return max(1, type_size)
    return None


def _vram_block_layout_enabled() -> bool:
    from runtime.env import vram_weight_block_layout_enabled

    return vram_weight_block_layout_enabled()


def ggml_tensor_storage_bytes(ne: int, type_id: int) -> int:
    """Bytes for ``ne`` elements (ggml block layout when known)."""
    if ne <= 0:
        return 0
    if _vram_block_layout_enabled():
        layout = _GGML_BLOCK_LAYOUT.get(type_id)
        if layout is not None:
            blck, type_size = layout
            if blck > 0 and ne % blck == 0:
                return (ne // blck) * type_size
    return ne * ggml_type_kv_bytes(type_id)


def ggml_type_weight_bytes(type_id: int) -> int:
    """Conservative per-element bytes when block layout is unknown."""
    return ggml_type_kv_bytes(type_id)


@dataclass(frozen=True)
class GgufTensorInfo:
    name: str
    nbytes: int
    layer: int | None = None
    ne: tuple[int, ...] = ()


def _tensor_layer_index(name: str) -> int | None:
    m = _BLK_LAYER_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def gguf_file_tensor_region_bytes(path: Path) -> int | None:
    """On-disk tensor payload bytes (includes inter-tensor alignment padding)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if _read_u32(f) != GGUF_MAGIC:
                return None
            version = _read_u32(f)
            if version == 1:
                return None
            n_tensors = _read_u64(f)
            n_kv = _read_u64(f)

            alignment = 32
            for _ in range(n_kv):
                key = _read_string(f)
                vtype = _read_u32(f)
                val = _read_gguf_value(f, vtype)
                if key == "general.alignment" and vtype == GGUF_TYPE_UINT32 and isinstance(val, int):
                    alignment = max(1, val)

            for _ in range(n_tensors):
                _read_string(f)
                ndim = _read_u32(f)
                f.seek(8 * ndim, 1)
                f.seek(4 + 8, 1)

            offset = f.tell()
            pad = (-offset) % alignment
            tensor_offset = offset + pad
            if tensor_offset >= size:
                return None
            return size - tensor_offset
    except (OSError, ValueError, struct.error, TypeError, EOFError):
        return None


def gguf_tensor_infos(path: Path) -> list[GgufTensorInfo] | None:
    """Per-tensor payload sizes from GGUF metadata."""
    try:
        with path.open("rb") as f:
            if _read_u32(f) != GGUF_MAGIC:
                return None
            version = _read_u32(f)
            if version == 1:
                return None
            n_tensors = _read_u64(f)
            n_kv = _read_u64(f)
            for _ in range(n_kv):
                _read_string(f)
                vtype = _read_u32(f)
                _skip_typed_value(f, vtype)
            out: list[GgufTensorInfo] = []
            for _ in range(n_tensors):
                name = _read_string(f)
                ndim = _read_u32(f)
                dims: list[int] = []
                ne = 1
                for _ in range(ndim):
                    d = int(_read_u64(f))
                    dims.append(d)
                    ne *= d
                type_id = _read_u32(f)
                _read_u64(f)
                nbytes = ggml_tensor_storage_bytes(int(ne), type_id)
                ne_tuple = tuple(dims) if len(dims) == 2 else ()
                out.append(
                    GgufTensorInfo(
                        name=name,
                        nbytes=nbytes,
                        layer=_tensor_layer_index(name),
                        ne=ne_tuple,
                    )
                )
            return out
    except (OSError, ValueError, struct.error, TypeError, EOFError):
        return None


def estimate_gpu_weight_bytes(path: Path, n_gpu_layers: int) -> int | None:
    """Sum weight tensor bytes expected on GPU for ``-ngl``."""
    tensors = gguf_tensor_infos(path)
    if tensors is None:
        return None
    file_bytes = gguf_file_tensor_region_bytes(path)

    def _full_payload() -> int:
        total = sum(t.nbytes for t in tensors)
        if file_bytes is not None:
            return max(total, file_bytes)
        return total

    if n_gpu_layers < 0:
        return _full_payload()
    if n_gpu_layers == 0:
        return 0
    partial = 0
    for t in tensors:
        if t.layer is not None:
            if t.layer < n_gpu_layers:
                partial += t.nbytes
        else:
            partial += t.nbytes
    return partial


def gguf_weight_bytes(path: Path) -> int | None:
    """Tensor payload bytes: max(metadata sum, on-disk region)."""
    tensors = gguf_tensor_infos(path)
    file_bytes = gguf_file_tensor_region_bytes(path)
    if tensors is not None:
        meta_sum = sum(t.nbytes for t in tensors)
        if file_bytes is not None:
            return max(meta_sum, file_bytes)
        return meta_sum
    return file_bytes
