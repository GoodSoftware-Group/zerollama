# h3-cuda lab

CUDA backend for [antirez/h3.c](https://github.com/antirez/h3.c) `h3_gpu.h`.

Canonical notes: zerollama [docs/h3-cuda-port.md](../../docs/h3-cuda-port.md).

**Lab path:** active build dir on CT 1564 is `/tmp/h3c-research/h3-cuda`; this `research/` tree is the durable snapshot (no `bin/`).

```bash
make all     # full smoke matrix
make dit     # BF16 synthetic DiT block
make dit8    # int8 synthetic DiT block
make int8    # quantize / linear / mlp / qkv int8
make bench8  # BF16 vs int8 linear microbench (cuBLAS path)
```

Requires CUDA (`ARCH=sm_120` default). Do **not** bind production ports `:11434` / `:8081`.

`h3_gpu_has_int8_mlp()` → 1 (portable int8 + cuBLAS when `K≥64`, `N≥16`, `K%4==0`).  
`h3_gpu_has_nax_mlp()` → 0 (Apple-only).  
`H3_DISABLE_CUBLAS_INT8=1` forces the shared-A CUDA kernel.
