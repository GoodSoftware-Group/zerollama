# CUDA UMA-toolkit twin (design)

**Why:** [`x/video-c`](../x/video-c/) (was `x/wan-c`) is Mac/`uma_daemon` today. Linux+CUDA CTs need the **same Pure-C Wan hotpath** without Darwin. This doc locks the twin so we ship **wins or unlocks**, not a second Gradio stack.

**Ownership:** CUDA lab targets (`make -C x/video-c cuda-lab`, rematch) are the parallel CUDA track. Darwin UMA + H3 `--info` — [video-c.md](./video-c.md).

**Status:** phase-1–2h green; **2e–2g** rematch + **2h** C `patch_embedding` (noise→tokens) wired into `cuda_latent_unipc_rematch` (all cosines ≈ 1.0). Not a product `/v1/videos` default.

## Win vs unlock matrix

| Work | Kind | Notes |
|------|------|-------|
| `dit_pager` N=2 | win + unlock | [dit-pager.md](./dit-pager.md) |
| Thin `wan_backend` vtable | unlock | Swap UMA ↔ CUDA without rewriting `wan_graph.c` forever |
| In-process CUDA cuBLAS GEMM | win thesis + unlock | No GRAPH IPC tax; sm_120 lab smoke |
| DiT FFN-linear fragment + pager | unlock proof | Gates FFN / block investment |
| **FFN half-block** (LN→AdaLN→up→GELU→down→gate) | **win + unlock** | `cuda_ffn_block`; elementwise CUDA kernels + pager on Wu/Wd |
| **Synthetic block0** (self-attn+FFN) | **unlock** | `cuda_block0`; RMS+RoPE3+SDPA+FFN; pager N=2 |
| **Real blocks.0..3** safetensors | **win + unlock** | `cuda_block0_real`; ~354 MiB pager peak w/ cross; CUDA ≫ host |
| **PyTorch block0 rematch** | **win** | `cuda_block0_rematch` + `gen_block0_cuda_fixture.py`; stage cosines ≈ 1.0 |
| **30-block trunk + UniPC step0** | **win + unlock** | `cuda_multiblock_rematch`; pager N=2 (~354 MiB); token-space UniPC cosine ≈ 1.0 |
| **Latent UniPC step0** | **win** | `cuda_latent_unipc_rematch` + `gen_latent_unipc_fixture.py`; head+unpatch → model_out → `sched_unipc`; cosine ≈ 1.0 |
| **C patch_embedding** | **win + unlock** | `wan_op_patch_embed_f32` (host Conv3d); noise→tokens in latent rematch cosine ≈ 1.0 |
| Product `ZEROLLAMA_VIDEO_CLI` Linux | next | Wire CUDA backend into `video-cli` / serve flag; lab ports only (`ZEROLLAMA_WAN_CLI` still accepted) |
| Full GRAPH string interpreter on CUDA | later | Prefer direct kernels for hot DiT first |
| Literal `mmgp` port | skip | Pager + kernels replace the idea |

## Process model (locked for phase 1)

| Choice | Decision |
|--------|----------|
| **In-process CUDA** | **Yes — first.** Link into lab binary / future `wan-cli` CUDA build. |
| Out-of-process daemon | **Design only.** Revisit if we need Mac-like cross-process buffer sharing with chat/serve. |
| Protocol | Named buffers (`buf_alloc` / `put` / `get` / `free`) + **direct** `gemm_f16` (cuBLAS). Optional GRAPH IR later. |
| Memory | `cudaMalloc` / host pinned H2D; BANK = persistent device blobs + bind aliases for resident DiT slots |
| Ports | Lab only — never `:11434` / `:8081` |
| CT 1564 `LD_LIBRARY_PATH` | `/root/nvidia-host:/usr/lib/ollama/cuda_v13:/usr/local/cuda/lib64` (`x/video-c` Makefile exports for `cuda-lab`) |

```text
wan-cli / lab smoke
    │
    ├─ dit_pager (portable)
    └─ wan_backend vtable
           ├─ backend_uma   (Darwin → uma_daemon)
           ├─ backend_cuda  (Linux in-process)
           └─ backend_host  (UMA_WAN_LOCAL / CPU fallback later)
```

## Op map (phase 1 minimal)

| Backend op | UMA today | CUDA phase 1 |
|------------|-----------|--------------|
| `create` / `destroy` | `uma_client_connect` | `cudaSetDevice` + cuBLAS handle |
| `buf_alloc` / `free` | `uma_buf_pool_*` | `cudaMalloc` / `cudaFree` |
| `buf_put` / `get` | BUF_PUT / BUF_GET | `cudaMemcpy` H2D / D2H |
| `bank_put` / `bind` / `evict` | BANK_* (+ bind); evict TBD on UMA | device blob + alias bind; **free on evict** |
| `layernorm` / `affine_mul_add` / `gelu_tanh` / `gated_residual` | GRAPH / host | CUDA kernels in `backend_cuda_elem.cu` |
| `head_rmsnorm` / `rope3` / `attn_sdpa` | GRAPH | CUDA RMS+SDPA; RoPE host-apply (Wan freqs) |
| `rmsnorm` / `bias_add` / `scale_bias` | host | WanRMSNorm + Linear bias + LN affine |
| `gemm_f16` | GRAPH `GEMM_F16` | cuBLAS (f32 lab; f16 next) |
| `sync` | wait ticket | `cudaDeviceSynchronize` |

Layer banks: key `W_L{id}`; on LRU eviction call `bank_evict` before the next `bank_put` (see [dit-pager.md](./dit-pager.md)).

## DiT fragment definition (concrete)

Lab smoke (`make -C x/video-c cuda-lab`):

- Synthetic **L=8** DiT layers; pager **N=2** (`WAN_DIT_RESIDENT=2`)
- Per layer weight: `W_L{i}` shape **[K=256, M=256]** f32; `bank_bind` → working `"W"`
- Activation `X` **[T=64, K=256]**; `Y = X @ W` via backend GEMM when layer touched
- Print: hits/misses, pager peak, **backend peak**, wall ms (GEMM smoke warms up before timing)

**Kill:** pager peak ≥ 80% of load-all weights, **or** backend peak > `N * W + X + Y` (means evict not wired).

## Relation to h3-cuda

Sibling research ([h3-cuda-port.md](./h3-cuda-port.md)). Borrow stream/kernel lessons; **do not** make MiniMax packs a dependency for Wan CUDA.

## Next win after patch embed

Product path: optional `ZEROLLAMA_VIDEO_CLI` / Linux `video-cli` CUDA backend (lab ports only). Lab rematch now owns **noise → patch → 30 blocks → head/unpatch → UniPC step0** end-to-end (`cuda_latent_unipc_rematch`).
