"""Native decode loop (Phase 15 v12–v15 — libllama link + C decode).

WHY this module exists before the loop ships: operators and CI need a stable probe
for whether ``llama_decode`` can run inside ``runtime.kv._kv_native`` vs ctypes.

v12: build-time libllama link + ``decode_loop_status`` probe.
v13: ``run_prefill`` and ``run_step`` call C implementations when the linked
     build is active, falling back to ctypes when not.
v14: GIL released during C ``llama_decode``; ``pos_start`` for remaining prefill;
     page-bind validation before C calls (matches ctypes batch path).
v15: ``run_sample`` + ``run_step(..., smpl_ptr=)`` call ``llama_sampler_sample`` in C.
v27: ``native_batch_decode_available()`` gates engine ``complete_parallel`` wiring.
v29: ``complete_parallel_stream`` + ``_decode_parallel_stream`` for tagged batch streaming.
v30: ``run_batch_step(..., smpl_ptrs=)`` — per-row C sampling after batch decode.
v31: ``prefill_abort_set`` / ``prefill_abort_clear`` — chunked-prefill ctx cancellation.
     ``run_prefill`` raises ``PrefillAbortedError`` when the C loop returns ERR_ABORT.
v32: ``ZEROLLAMA_KV_AUTO_BATCH=1`` — opt-in coordinator coalesces concurrent
     ``generate()`` into ``completions_parallel`` (in-process multiseq only).

``run_prefill`` and ``run_step`` accept a raw ``llama_context`` pointer as a
Python int (the value returned by ctypes ``c_void_p``).  The C layer casts it
to ``struct llama_context *`` directly — no ABI mismatch because the same
libllama is loaded by both paths.
"""

from __future__ import annotations

import os
from typing import Any


class PrefillAbortedError(RuntimeError):
    """Raised when a chunked prefill was cancelled via ``prefill_abort_set``.

    WHY separate exception class: callers (engine, greedy_decode_tokens) need to
    distinguish abort from llama_decode failure (-1) and page-bind failure (-2)
    without inspecting string messages.
    """


def native_decode_loop_available() -> bool:
    return decode_loop_status().get("available") is True


def native_batch_decode_available() -> bool:
    """True when v26 batch decode is linked and not disabled by env (v27)."""
    from runtime.env import kv_native_batch_enabled, kv_native_decode_enabled

    if not kv_native_batch_enabled():
        return False
    if not kv_native_decode_enabled():
        return False
    st = decode_loop_status()
    return bool(st.get("available") and st.get("batch_decode_in_c"))


def decode_loop_status() -> dict[str, Any]:
    """Return ``{available, reason, link, gil_released?, sampling_in_c?}``."""
    try:
        from runtime.kv._kv_native import decode_loop_status as _status

        raw = _status()
        if isinstance(raw, dict):
            out: dict[str, Any] = {
                "available": bool(raw.get("available")),
                "reason": str(raw.get("reason") or ""),
                "link": str(raw.get("link") or "ctypes"),
            }
            if raw.get("llama_max_devices") is not None:
                out["llama_max_devices"] = int(raw["llama_max_devices"])
            if raw.get("gil_released") is not None:
                out["gil_released"] = bool(raw["gil_released"])
            if raw.get("sampling_in_c") is not None:
                out["sampling_in_c"] = bool(raw["sampling_in_c"])
            if raw.get("batch_decode_in_c") is not None:
                out["batch_decode_in_c"] = bool(raw["batch_decode_in_c"])
            return out
    except ImportError:
        pass
    except Exception:
        pass
    return {
        "available": False,
        "reason": "runtime.kv._kv_native not built",
        "link": "ctypes",
    }


def prefill_abort_set() -> None:
    """Signal the C prefill loop to abort after the current chunk (v31).

    Safe to call from any Python thread while ``run_prefill`` holds the GIL
    released.  The C atomic write is visible to the prefill thread before the
    next chunk boundary check.

    After the aborted prefill raises ``PrefillAbortedError``, the caller must
    call ``prefill_abort_clear()`` before the next ``run_prefill``.
    """
    try:
        from runtime.kv._kv_native import decode_loop_abort_set

        decode_loop_abort_set()
    except ImportError:
        pass


def prefill_abort_clear() -> None:
    """Reset the C prefill abort flag (v31).

    Must be called before the next ``run_prefill`` when a previous call was
    aborted, or before arming a new cancellation timeout.
    """
    try:
        from runtime.kv._kv_native import decode_loop_abort_clear

        decode_loop_abort_clear()
    except ImportError:
        pass


def run_prefill(
    ctx_ptr: int,
    tokens: list[int],
    *,
    seq_id: int = 0,
    block_size: int = 0,
    pos_start: int = 0,
    kv_slot: int | None = None,
) -> int | None:
    """Call ``kv_decode_loop_run_prefill`` in C; return step count or None if not linked.

    WHY page-bind check here: ctypes path validates in ``build_batch_from_tokens``;
    C path must fail fast with the same ``LlamaServerError`` before touching llama.

    WHY abort_clear before call: ensures a stale abort flag from a previous
    cancelled request does not immediately abort the new prefill (v31).

    Raises ``PrefillAbortedError`` when the C loop returns KV_DECODE_LOOP_ERR_ABORT
    (-3), i.e. ``prefill_abort_set()`` was called from another thread between chunks.
    """
    if not native_decode_loop_available():
        return None
    if kv_slot is not None and tokens:
        from runtime.kv.page_bind import validate_token_positions

        validate_token_positions(int(kv_slot), int(pos_start), len(tokens))
    prefill_abort_clear()
    try:
        from runtime.kv._kv_native import decode_loop_prefill

        return int(
            decode_loop_prefill(
                ctx_ptr, tokens, seq_id, block_size, int(pos_start)
            )
        )
    except ValueError as e:
        msg = str(e)
        if "KV prefill aborted" in msg:
            raise PrefillAbortedError(msg) from e
        if "KV page bind" in msg:
            from runtime.worker.llama_server import LlamaServerError

            raise LlamaServerError(msg) from e
        raise


def run_batch_step(
    ctx_ptr: int,
    tokens: list[int],
    seq_ids: list[int],
    positions: list[int],
    *,
    smpl_ptr: int = 0,
    smpl_ptrs: list[int] | None = None,
) -> int | tuple[int, list[int]] | None:
    """Run one continuous-batch decode step in C (v26/v30).

    One ``llama_decode`` for N single-token rows (one per active sequence).
    When ``smpl_ptrs`` is set (v30), each row uses its own sampler with the
    correct logit index. ``smpl_ptr`` (legacy int) repeats one sampler for all
    rows — only safe when ``len(tokens)==1``.
    Returns ``None`` when the linked ext is not built.
    """
    if not native_decode_loop_available():
        return None
    if not (len(tokens) == len(seq_ids) == len(positions)):
        raise ValueError("tokens, seq_ids, positions length mismatch")
    if not tokens:
        return 0
    try:
        from runtime.kv._kv_native import decode_loop_batch_step

        smpl_arg: int | list[int] | None = None
        if smpl_ptrs is not None:
            smpl_arg = smpl_ptrs
        elif smpl_ptr:
            smpl_arg = int(smpl_ptr)
        if smpl_arg is not None:
            steps, sampled = decode_loop_batch_step(
                ctx_ptr, tokens, seq_ids, positions, smpl_arg
            )
            return int(steps), [int(x) for x in sampled]
        return int(decode_loop_batch_step(ctx_ptr, tokens, seq_ids, positions))
    except ValueError as e:
        from runtime.worker.llama_server import LlamaServerError

        if "KV page bind" in str(e):
            raise LlamaServerError(str(e)) from e
        raise


def run_sample(smpl_ptr: int, ctx_ptr: int) -> int | None:
    """Sample via C ``llama_sampler_sample``; None when ext not linked."""
    if not native_decode_loop_available() or not smpl_ptr or not ctx_ptr:
        return None
    from runtime.kv._kv_native import decode_loop_sample

    return int(decode_loop_sample(int(smpl_ptr), int(ctx_ptr)))


def run_step(
    ctx_ptr: int,
    token: int,
    *,
    seq_id: int = 0,
    current_pos: int = 0,
    kv_slot: int | None = None,
    smpl_ptr: int = 0,
) -> int | tuple[int, int] | None:
    """Call ``kv_decode_loop_run_step`` in C.

    When ``smpl_ptr`` is set, returns ``(steps, sampled_token)`` (v15).
    """
    if not native_decode_loop_available():
        return None
    if kv_slot is not None:
        from runtime.kv.page_bind import validate_token_positions

        validate_token_positions(int(kv_slot), int(current_pos), 1)
    try:
        from runtime.kv._kv_native import decode_loop_step

        if smpl_ptr:
            steps, sampled = decode_loop_step(
                ctx_ptr, token, seq_id, current_pos, int(smpl_ptr)
            )
            return int(steps), int(sampled)
        return int(decode_loop_step(ctx_ptr, token, seq_id, current_pos))
    except ValueError as e:
        from runtime.worker.llama_server import LlamaServerError

        if "KV page bind" in str(e):
            raise LlamaServerError(str(e)) from e
        raise


def greedy_decode_tokens(
    ctx_ptr: int,
    lib: Any,
    ctx: Any,
    vocab: Any,
    smpl: Any,
    prompt_tokens: list[int],
    *,
    n_predict: int,
    seq_id: int = 0,
    block_size: int = 0,
    pos_start: int = 0,
    kv_slot: int | None = None,
) -> list[int]:
    """Greedy decode via C prefill + C steps + C/ctypes sampling (E2E / parity tests).

    WHY separate helper: lets E2E smoke compare token ids against ctypes ``_decode_stream``
    without duplicating the autoregressive loop in every test.
    """
    import ctypes

    if not native_decode_loop_available():
        raise RuntimeError("native decode loop not linked")
    steps = run_prefill(
        ctx_ptr,
        prompt_tokens,
        seq_id=seq_id,
        block_size=block_size,
        pos_start=pos_start,
        kv_slot=kv_slot,
    )
    if steps is None:
        raise RuntimeError("run_prefill returned None despite linked build")

    n_prompt = len(prompt_tokens)
    n_pos = pos_start + n_prompt
    out: list[int] = []
    limit = max(0, int(n_predict))
    if limit == 0:
        return out

    smpl_int = int(ctypes.cast(smpl, ctypes.c_void_p).value or 0)
    use_c_sample = bool(smpl_int)

    def _sample_token() -> int:
        if use_c_sample:
            tid = run_sample(smpl_int, ctx_ptr)
            if tid is not None:
                return int(tid)
        return int(lib.llama_sampler_sample(smpl, ctx, -1))

    new_id = _sample_token()
    if lib.llama_vocab_is_eog(vocab, new_id):
        return out
    out.append(new_id)

    while len(out) < limit:
        if use_c_sample:
            step_out = run_step(
                ctx_ptr,
                new_id,
                seq_id=seq_id,
                current_pos=n_pos,
                kv_slot=kv_slot,
                smpl_ptr=smpl_int,
            )
            if not isinstance(step_out, tuple):
                raise RuntimeError("run_step expected (steps, token) with smpl_ptr")
            n_pos += 1
            new_id = int(step_out[1])
        else:
            step = run_step(
                ctx_ptr, new_id, seq_id=seq_id, current_pos=n_pos, kv_slot=kv_slot
            )
            if step is None:
                raise RuntimeError("run_step returned None despite linked build")
            n_pos += 1
            new_id = _sample_token()
        if lib.llama_vocab_is_eog(vocab, new_id):
            break
        out.append(new_id)
        if len(out) >= limit:
            break
    return out
