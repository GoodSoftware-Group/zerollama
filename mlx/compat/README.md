# MLX-C carry patches

Patches under `mlx-c/` are applied at FetchContent time via
`apply-git-patches.cmake` (see `x/mlxrunner/mlx/CMakeLists.txt` and
`x/imagegen/mlx/CMakeLists.txt`).

When `OLLAMA_MLX_C_SOURCE` points at a sibling checkout (the usual Mac
lab path), FetchContent skips `PATCH_COMMAND` — `scripts/mlx/ensure_mlx_sources.sh`
re-applies these patches onto that tree before `build_mlx_dylibs_mac.sh`.

| Patch | Why |
|-------|-----|
| `0001-mlx-c-qmm-global-scale.patch` | `gather_qmm` optional `global_scale` (upstream Ollama) |
| `0002-fast-gated-delta-update.patch` | `mlx_fast_gated_delta_update` until mlx-c ships it |

`0001-mlx-c-regen-0.32.1.patch` (repo root of this dir) is **obsolete** on
`MLX_C_VERSION` ≥ `ebc88f10` — do not apply.

**Metal idle residency:** Ollama briefly carried `mlx/compat/mlx/0001-metal-residency-refresh.patch` (#18807); at **`MLX_VERSION=a59cc231`** the fix is **upstream in mlx** (#18854 dropped the carry patch). Rebuild dylibs after pin bump; the Go client sets `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` when unset.
