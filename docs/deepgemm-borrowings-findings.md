# DeepGEMM borrowings — rematch findings

**Spec:** [deepgemm-borrowings.md](./deepgemm-borrowings.md)  
**Rematch:** Oct 2026 (README tip + news through 2026-09-30)  
**Non-goal reminder:** do **not** vendor DeepGEMM / DeepGEMM-Ascend / DeepJIT into serve.

## Hardware mismatch (why “ideas only”)

| Stack | Arch DeepGEMM advertises | Our lanes |
|-------|--------------------------|-----------|
| DeepGEMM CUDA | **SM90 / SM100** (Hopper / Blackwell datacenter class in their docs) | **sm_89** (4090), **sm_120** (5080) via ggml |
| DeepGEMM-Ascend | Ascend 950 / CANN | **None** |

We do not claim DeepGEMM tok/s on 5080/4090. Delivery stays **ggml `mmq` / fork kernels** ([cuda-lanes.md](./cuda-lanes.md) P1 done on dual-4090; P2 Blackwell MMA still 5080-only).

## Rematch map (Oct 2026 tip)

| DeepGEMM surface (README) | Zerollama action |
|---------------------------|------------------|
| Dense `fp8_gemm_*` / `fp8_fp4_gemm_*` + SF layouts (UE8M0 on SM100) | **Already covered directionally** — GGUF NVFP4 / MXFP4 / FP8; SF as load concern. No Python DeepGEMM. |
| `m_grouped_*_contiguous` + `get_mk_alignment_for_contiguous_layout()` | **Watch** — document expert M-alignment when MoE smokes mis-pack; do not port DeepGEMM API |
| `m_grouped_*_masked` (decode + CUDA graph, DeepEP-shaped) | **Watch** — ggml decode graphs + MoE paths; no DeepEP dependency ([deepinfer-ladder.md](./deepinfer-ladder.md) also skips DeepEP) |
| Mega MoE (`fp8_fp4_mega_moe`, EP + symm memory, PyTorch ≥2.9) | **Skip product** — multi-rank EP / NVLink fuse; Flash-MoE is SSD residency, not this |
| Lightning indexer MQA logits (`fp8_fp4_mqa_logits` / paged) | **Skip** until a DeepSeek indexer GGUF is a product target |
| Mega mHC / HyperConnection prenorm | **Skip** unless GGUF graphs require that family |
| DeepJIT disk cache (`DG_JIT_CACHE_DIR`) | **Watch habit only** — ggml CUDA graphs already cache; no second JIT runtime |
| Ascend port (2026-09-30 news) | **Ignore for product** — checklist only |

## Decisions

| Choice | WHY |
|--------|-----|
| Keep borrowings doc; no ROADMAP “Mxx vendor DeepGEMM” | Arch + EP + JIT stack do not fit single-GPU ggml serve |
| Findings file + host unit | Agents rematch without cloning; CI can assert non-goals stay documented |
| Point operators at cuda-lanes P2 for Blackwell wins | Real 5080 work is ggml `blackwell_mma_available()`, not `pip install deep_gemm` |

## Checklist

- [x] README tip skim (grouped contiguous/masked, Mega MoE, indexer, Ascend news)
- [x] Map to Taken / Watch / Skip (above)
- [x] Confirm no in-tree `deep_gemm` / Ascend imports in runtime
- [ ] Optional: clone `../DeepGEMM` on Mac lab when reading kernel sources (headers only)

## Host unit

```bash
./scripts/phase/deepgemm_borrowings_unit.sh
```
