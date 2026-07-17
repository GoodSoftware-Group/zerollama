"""Parse llama-server CLI flags for VRAM estimates (Phase 13 flag parity)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaServerArgHints:
    """Subset of llama-server flags that affect VRAM budgeting and in-process load."""

    num_ctx: int | None = None
    n_gpu_layers: int | None = None
    parallel_slots: int | None = None
    main_gpu: int | None = None
    split_mode: str | None = None
    tensor_split: tuple[float, ...] | None = None
    draft_model: str | None = None
    draft_n_gpu_layers: int | None = None
    spec_type: str | None = None


_CTX_FLAGS = frozenset({"-c", "--ctx-size", "--ctx_size"})
_MG_FLAGS = frozenset({"-mg", "--main-gpu", "--main_gpu"})
_SM_FLAGS = frozenset({"-sm", "--split-mode", "--split_mode"})
_TS_FLAGS = frozenset({"-ts", "--tensor-split", "--tensor_split"})
_SPEC_TYPE_FLAGS = frozenset({"--spec-type", "--spec_type"})
_NGL_FLAGS = frozenset({"-ngl", "--n-gpu-layers", "--n_gpu_layers"})
_NP_FLAGS = frozenset({"-np", "--parallel", "--parallel-slots", "--parallel_slots"})
_DRAFT_MODEL_FLAGS = frozenset({"--model-draft", "--model_draft"})
_DRAFT_NGL_FLAGS = frozenset({"--spec-draft-ngl", "--spec_draft_ngl"})


def _read_int(argv: list[str], i: int) -> tuple[int | None, int]:
    if i + 1 >= len(argv):
        return None, i
    try:
        return int(argv[i + 1]), i + 1
    except ValueError:
        return None, i


def _read_eq_int(token: str) -> int | None:
    if "=" not in token:
        return None
    _, raw = token.split("=", 1)
    try:
        return int(raw)
    except ValueError:
        return None


def parse_llama_server_args(argv: list[str] | None) -> LlamaServerArgHints:
    """Extract VRAM-relevant flags from a llama-server argv tail (last value wins)."""
    if not argv:
        return LlamaServerArgHints()
    num_ctx: int | None = None
    n_gpu_layers: int | None = None
    parallel_slots: int | None = None
    draft_model: str | None = None
    draft_n_gpu_layers: int | None = None
    spec_type: str | None = None
    main_gpu: int | None = None
    split_mode: str | None = None
    tensor_split: tuple[float, ...] | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        eq = _read_eq_int(tok)
        if eq is not None:
            base = tok.split("=", 1)[0]
            if base in _CTX_FLAGS:
                num_ctx = eq
            elif base in _NGL_FLAGS:
                n_gpu_layers = eq
            elif base in _NP_FLAGS:
                parallel_slots = eq
            elif base in _DRAFT_NGL_FLAGS:
                draft_n_gpu_layers = eq
            elif base in _MG_FLAGS:
                main_gpu = eq
            i += 1
            continue
        if "=" in tok and tok.split("=", 1)[0] in _TS_FLAGS:
            raw_ts = tok.split("=", 1)[1]
            tensor_split = _parse_tensor_split(raw_ts)
            i += 1
            continue
        if "=" in tok and tok.split("=", 1)[0] in _SPEC_TYPE_FLAGS:
            spec_type = tok.split("=", 1)[1].strip().lower() or None
            i += 1
            continue
        if tok in _CTX_FLAGS:
            v, i = _read_int(argv, i)
            if v is not None and v > 0:
                num_ctx = v
        elif tok in _NGL_FLAGS:
            v, i = _read_int(argv, i)
            if v is not None:
                n_gpu_layers = v
        elif tok in _NP_FLAGS:
            v, i = _read_int(argv, i)
            if v is not None and v > 0:
                parallel_slots = v
        elif tok in _DRAFT_MODEL_FLAGS:
            if i + 1 < len(argv):
                draft_model = argv[i + 1]
                i += 1
        elif tok in _DRAFT_NGL_FLAGS:
            v, i = _read_int(argv, i)
            if v is not None:
                draft_n_gpu_layers = v
        elif tok in _SPEC_TYPE_FLAGS:
            if i + 1 < len(argv):
                spec_type = argv[i + 1].strip().lower() or None
                i += 1
        elif tok in _MG_FLAGS:
            v, i = _read_int(argv, i)
            if v is not None:
                main_gpu = v
        elif tok in _SM_FLAGS:
            if i + 1 < len(argv):
                split_mode = argv[i + 1].strip().lower() or None
                i += 1
        elif tok in _TS_FLAGS:
            if i + 1 < len(argv):
                tensor_split = _parse_tensor_split(argv[i + 1])
                i += 1
        i += 1
    return LlamaServerArgHints(
        num_ctx=num_ctx,
        n_gpu_layers=n_gpu_layers,
        parallel_slots=parallel_slots,
        main_gpu=main_gpu,
        split_mode=split_mode,
        tensor_split=tensor_split,
        draft_model=draft_model,
        draft_n_gpu_layers=draft_n_gpu_layers,
        spec_type=spec_type,
    )


def resolve_parallel_slots(
    llama_args: list[str] | None,
    *,
    default: int = 1,
) -> int:
    """Effective ``-np`` for slot allocator + in-process ``n_seq_max`` (argv wins over YAML default)."""
    slots = parse_llama_server_args(llama_args).parallel_slots
    if slots is not None and slots > 0:
        return slots
    return max(1, int(default))


def _parse_tensor_split(raw: str) -> tuple[float, ...] | None:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return None


def split_mode_to_llama_cpp_int(split_mode: str | None) -> int:
    """Map llama-server split-mode strings to llama_cpp.LLAMA_SPLIT_MODE_* ints."""
    import llama_cpp

    sm = (split_mode or "layer").strip().lower()
    table = {
        "none": llama_cpp.LLAMA_SPLIT_MODE_NONE,
        "layer": llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        "row": llama_cpp.LLAMA_SPLIT_MODE_ROW,
        "tensor": llama_cpp.LLAMA_SPLIT_MODE_TENSOR,
    }
    return table.get(sm, llama_cpp.LLAMA_SPLIT_MODE_LAYER)


def inprocess_speculative_requested(hints: LlamaServerArgHints) -> bool:
    if hints.draft_model:
        return True
    st = (hints.spec_type or "").strip().lower()
    return bool(st and st != "none")


def with_llama_num_ctx(argv: list[str], num_ctx: int) -> list[str]:
    """Copy argv with ``-c`` / ``--ctx-size`` set to ``num_ctx`` (for load/estimate parity)."""
    if num_ctx <= 0:
        return list(argv)
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        base = tok.split("=", 1)[0]
        if base in _CTX_FLAGS:
            if "=" in tok:
                out.append(f"{base}={num_ctx}")
            else:
                out.extend([tok, str(num_ctx)])
                if i + 1 < len(argv):
                    i += 1
            i += 1
            continue
        out.append(tok)
        i += 1
    if parse_llama_server_args(out).num_ctx != num_ctx:
        out.extend(["-c", str(num_ctx)])
    return out


_KV_UNIFIED_ON = frozenset({"-kvu", "--kv-unified"})
_KV_UNIFIED_OFF = frozenset({"-no-kvu", "--no-kv-unified"})


def _argv_has_flag(argv: list[str], flags: frozenset[str]) -> bool:
    for tok in argv:
        base = tok.split("=", 1)[0]
        if base in flags:
            return True
    return False


def with_llama_kv_unified(argv: list[str], enabled: bool) -> list[str]:
    """Phase 15 v53: inject ``--kv-unified`` for subprocess when opted in.

    WHY: v52 only set ``cparams.kv_unified`` for in-process; L3 agent / Radix
    live path is subprocess. Operator ``--no-kv-unified`` / ``-no-kvu`` in
    ``LLAMA_SERVER_EXTRA_ARGS`` wins (explicit override). When ``enabled`` is
    false, existing flags are left alone.
    """
    out = list(argv)
    if not enabled:
        return out
    if _argv_has_flag(out, _KV_UNIFIED_OFF):
        return out
    if _argv_has_flag(out, _KV_UNIFIED_ON):
        return out
    out.append("--kv-unified")
    return out
