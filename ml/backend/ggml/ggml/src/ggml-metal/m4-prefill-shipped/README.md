# m4-prefill-shipped

Apache-2.0 kernels from [m4-prefill-engine](https://github.com/mohamedhossammohamed/m4-prefill-engine)
(Copyright 2026 Mohammed Hossam). See `LICENSE`.

| File | Priority | Status |
|------|----------|--------|
| `m4_fused_swiglu.metal` | P0 | **Wired** as embed kind `m4_fused_swiglu` (`ZEROLLAMA_M4_PREFILL_SWIGLU=1`) |
| `fused_gate_up_swiglu_q4_0.metal` | P0 | Upstream-shaped reference (kept) |
| `quantize_kv_to_q8_0.metal` | P1 | reference only — stock `SET_ROWS` writes Q8 |
| `flash_attn_q8_0_causal.metal` | P1 | reference only — prefer stock FA + `q8_0` KV (kv_f16); `Q8_KV` native skip is experimental / slower on M4 Max |
| `pipe_qkv_head_gemm_q4_0.metal` | P2 | staged |
| `pipe_gemm_q4_0_32x32.metal` | P3 | staged |
| `swiglu_activation.metal` | ref | staged |

Guide: [docs/m4-prefill-borrowings.md](../../../../../../../docs/m4-prefill-borrowings.md).
Sibling: `../m4-prefill-engine`.
