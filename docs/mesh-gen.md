# Local GLB generation (TRELLIS lab)

**Why this track:** Chat and video already share the GPU job queue. A room prop is a **textured PBR GLB** (metallic-roughness), not a chat completion and not a Gaussian ply alone.

**Status:** Lab finish line on astra (CUDA) and on the M4 (Metal). Astra artifacts under `/mnt/ssd2/zerollama/third_party/trellis/lab/out/`. Mac artifacts under `~/.zerollama/third_party/trellis-mac/lab/out/` (`trellis2-mps.glb`). **Service shape:** image still → rembg → TRELLIS.2 (PBR). Text XL stays a compare / no-imagegen fallback. Next slices: license-clean bake, `POST /v1/meshes`, lit Mac preview.

## Machines

| Host | Role |
|------|------|
| **M4 Max (128 GB)** | Object still via existing MLX image gen + `rembg`, **and** TRELLIS.2 via Metal (`mtlgemm` / `mtlmesh` / `mtlbvh` / `mtldiffrast` through [trellis-mac](https://github.com/shivampkumar/trellis-mac)). |
| **astra (2× RTX 4090)** | CUDA TRELLIS.2 reference (+ optional text XL). One process → **one card** (24 GB). Default `CUDA_VISIBLE_DEVICES=1` when GPU 0 serves. |

Do not start this inside `zerollama.service` (astra has already OOM-killed serve that way). Do not bind production `:11434` / `:8081` / Proxmox host serve for this lab. Keep Metal / CUDA venvs off the serve `PYTHONPATH`.

## Texture path (image gen → TRELLIS.2)

TRELLIS.2 is **image-to-3D**. It builds the mesh and the PBR volume (base color, metal, roughness, opacity) from one still, then the bake writes glTF materials.

**Mac (local finish):**

1. Prompt an image model as **one object on a plain background** (existing serve; do not start one for this lab).
2. `rembg` → RGBA PNG (`mac_object_still.sh`).
3. `trellis_mesh_generate_mac.py --resolution 512` (Metal baker). Hole fill is off on Darwin (`mtlmesh` crashes on decoder-sized meshes). RMBG-2.0 is **not** downloaded when the still already has alpha.

**astra (CUDA reference):**

1–2. Same still (or copy the PNG from the M4).
3. `trellis_mesh_generate.py --resolution 1024` on the free 4090.

A direct photo upload skips the imagegen step and still hits TRELLIS.2. This is **not** Hunyuan’s separate paint stage on an existing mesh.

Text XL takes the prompt directly and emits a **color** mesh (no metal/roughness maps). It replaces the image step only if that color mesh is acceptable **in the room** under real lights. Text XL is astra-only for now.

## Install — Mac (Metal)

```bash
# on the M4
cd ~/Sites/inference/zerollama   # or your clone
./scripts/mesh/install_trellis_mac.sh
source ~/.zerollama/third_party/trellis-mac/env-mac.sh
```

Pins [shivampkumar/trellis-mac](https://github.com/shivampkumar/trellis-mac), builds Pedro’s `mtlbvh` / `mtldiffrast` / `mtlmesh` / `mtlgemm` plus the Apple `o_voxel` fork, applies `patches/mps_compat.py`, and **fails** unless `o_voxel.postprocess` reports `_BACKEND=metal` with `_HAS_DR` and `_HAS_FLEX_GEMM`. Torch 2.15 ATen needs C++20 (install patches the Metal `setup.py` files). Eigen is pulled for `o-voxel`. Weights land in `$TRELLIS_ROOT/hf-cache`. Do not require `hf auth login` for the acceptance still (DINOv3 camenduru remap + rembg stub). A KDTree bake (`SKIP_METAL=1`) is not the acceptance path.

**Mac lab gotchas:**

| Issue | Lab fix |
|-------|---------|
| Sparse attention on MPS | Keep `ATTN_BACKEND=sdpa`. Microbench (`scripts/mesh/bench_mps_sparse_attn.py`) uses TRELLIS.2-4B shapes: structure `T=4096` (`16³`), `H=12`, `head_dim=128`; SLat same `H`/`C` with sparse `T` stand-in. Fused loses to native MPS SDPA on N=1. Vendored `deps/mtlgemm/tests/bench_sparse_attn.py` still times a CPU-bounce SDPA baseline — ignore that “5–15×”. Artifact: `lab/out/attn_microbench.json` |
| Cold start dominated by weight load | Lab resident process loads TRELLIS.2 once: `./scripts/mesh/mac_mesh_lab.sh` → `127.0.0.1:18091` JSONL (jobs serialized). Not a zerollama sidecar; never `:11434` / `:8081`. `cmd:shutdown` stops the process; `quit` drops one client. |
| `segment_reduce` crash | `PYTORCH_ENABLE_MPS_FALLBACK=1` **before** importing torch |
| Decode-time hole fill | Stays stubbed (`decode_hole_fill=skipped`). trellis-mac skips `Mesh.fill_holes`, `remove_faces`, and `simplify` on the decode mesh because that path segfaulted on large meshes. Bounds checks inside `MtlMesh::fill_holes` do not cover the preamble. Leave the stub. |
| Bake-time hole fill | Non-remesh `to_glb` calls `CuMesh.fill_holes` three times. Each call adds no faces when no boundary loop has perimeter ≤ 0.03, and it does not weld face-connected components. Meshes over 200k faces are `fast_simplification`-capped first, so the first fill sees that reduced mesh. `--skip-bake` records `bake_hole_fill=not_run`. |
| Metal BVH on huge meshes | Pre-simplify ≤200k faces, then bake `decimation_target≈100k` |
| Empty mesh / GPU watchdog | One retry with `MTL_CAPTURE_ENABLED=1`; do not accept KDTree |
| Shattered mesh (many components) | Photo mug raw bake: 93 components. Prune: position-weld coincident verts (one UV kept at seams) → drop a sheet only when both long axes are `>0.9` and the short axis is `<0.02`, and a piece with min extent `≥0.02` remains (a lone flat object stays) → `--prune-min-faces 100 --keep-largest 8`. Component face counts are a partition (face adjacency, not `trimesh.split`). Welded copy: 8→2, body largest by extent, PNG PBR kept. The leftover slab stays in that file. Fail if still >50 comps. |
| Open body (not watertight) | Boundary loops on the welded body are large and non-planar, so a fan cap cuts through the mug and `fill_holes` adds almost nothing. `--from-glb` writes a separate voxel crust of the largest component (`scikit-image` marching cubes, pitch = longest axis / `--shell-divisions`, default 56). It is a thin closed skin, not a filled solid. UVs are copied from the nearest source vertex; the PNG is not re-baked. Source GLB is left in place. |
| Floating / clipart still | Bad input — use a photo (`still_mug_photo.png`) or `make_connected_still.py` |
| Floating Metal dep HEAD | Install pins known-good SHAs (`TRELLIS_DEP_*`) into `env-mac.sh` |
| Gated DINOv3 / RMBG-2.0 | Same as astra: camenduru remap + stub rembg when still has alpha |

## Install — astra (sm_89 — not Blackwell wheels)

Prebuilt TRELLIS.2 / Comfy wheels are often **sm_120 only**. They fail on Ada with “no kernel image”.

```bash
# on astra (SSH host may be named cozmic)
cd ~/zerollama
export TRELLIS_ROOT=/mnt/ssd2/zerollama/third_party/trellis
# Prefer system nvcc; do not point CUDA_HOME at a broken cuda-shim.
# If host toolkit is 13.x and cu124 extension builds fail:
#   TRELLIS_TORCH_INDEX=cu130 …
CUDA_VISIBLE_DEVICES=1 ./scripts/mesh/install_trellis_astra.sh
# optional compare half (separate venv):
CUDA_VISIBLE_DEVICES=1 ./scripts/mesh/install_trellis_astra.sh --with-text
source "$TRELLIS_ROOT/env.sh"
# optional convenience link:
mkdir -p ~/.zerollama/third_party && ln -sfn "$TRELLIS_ROOT" ~/.zerollama/third_party/trellis
```

Builds `cumesh`, `flex_gemm`, `o-voxel` (and lab-only `nvdiffrast`) with `TORCH_CUDA_ARCH_LIST=8.9`, C++20, and a `sitecustomize` that skips torch↔nvcc version mismatch checks. `nvdiffrec` is **not** installed by default (GLB bake does not need it; pass `--with-nvdiffrec` only if you want it). Weights land in `$TRELLIS_ROOT/hf-cache`. Keep that venv off the serve library path and out of git.

**Ada lab gotchas (baked into the install / wrapper):**

| Issue | Lab fix |
|-------|---------|
| Prebuilt sm_120 wheels | Build extensions with `TORCH_CUDA_ARCH_LIST=8.9` |
| Broken `/mnt/ssd2/cuda-shim-*` | Prefer system `/usr/local/cuda*` or unset `CUDA_HOME` |
| ATen needs C++20 | Patch extension `setup.py` `c++17→c++20`; `CXXFLAGS=-std=c++20` |
| Gated `facebook/dinov3-…` | Wrapper remaps to `camenduru/dinov3-…` (same weights; DINOv3 license still applies) via `DINOV3_MODEL` |
| Gated `briaai/RMBG-2.0` | If the still already has alpha, rembg is **stubbed** (no download). `--preprocess-image` remaps to `ZhengPeng7/BiRefNet` — a **different** model, not a weight mirror |
| transformers≥5 nested DINOv3 encoder | Wrapper patches `extract_features` → `model.model.layer` |
| Sparse attention needs flash_attn/xformers | Default `ATTN_BACKEND=xformers` (sdpa is dense-only) |
| 1024 cascade OOM on 24 GB | Wrapper defaults `--low-vram` for resolution≥1024 on ≤24 GB cards |
| PATH putting `/usr/bin` before the venv | Install / compare scripts keep `$TRELLIS_VENV/bin` first |

## Commands

```bash
# Mac — still + Metal TRELLIS.2 (serve optional; pass --image if down)
source ~/.zerollama/third_party/trellis-mac/env-mac.sh
# Prefer a rembg'd photo of one object (clipart stills shatter):
python scripts/mesh/trellis_mesh_generate_mac.py \
  --image /tmp/trellis-lab/still_mug_photo.png \
  --out ~/.zerollama/third_party/trellis-mac/lab/out/trellis2-mps.glb \
  --resolution 512
python3 scripts/mesh/test_glb_inspect.py --require-pbr --require-png \
  ~/.zerollama/third_party/trellis-mac/lab/out/trellis2-mps.glb
# Optional one-shot when a serve is already listening:
#   ./scripts/mesh/mac_mesh.sh --prompt "a red ceramic mug, product photo"

# Mac — attention microbench (no GLB; tens of seconds)
python scripts/mesh/bench_mps_sparse_attn.py
# → lab/out/attn_microbench.json  (winner should be sdpa)

# Mac — resident lab process (load once; bake many stills)
./scripts/mesh/mac_mesh_lab.sh          # 127.0.0.1:18091
printf '%s\n' '{"cmd":"ping"}' | nc 127.0.0.1 18091
printf '%s\n' '{"image":"/tmp/trellis-lab/still_mug_photo.png","out":"'"$HOME"'/.zerollama/third_party/trellis-mac/lab/out/resident.glb","seed":1}' \
  | nc 127.0.0.1 18091

# astra — CUDA TRELLIS.2 (+ text XL if installed)
./scripts/mesh/mac_object_still.sh \
  --prompt "a red ceramic mug, product photo" \
  --out /tmp/still.png
scp /tmp/still.png cozmic:/mnt/ssd2/zerollama/third_party/trellis/lab/still.png
ssh cozmic
source /mnt/ssd2/zerollama/third_party/trellis/env.sh
cd /path/to/zerollama
./scripts/mesh/compare_trellis_prompt.sh \
  --prompt "a red ceramic mug, product photo" \
  --image /mnt/ssd2/zerollama/third_party/trellis/lab/still.png \
  --still-source "mlx z-image-turbo + rembg on M4" \
  --out-dir /mnt/ssd2/zerollama/third_party/trellis/lab/out
```

Inspect acceptance:

```bash
python3 scripts/mesh/test_glb_inspect.py --require-pbr /path/to/trellis2.glb
# no args → synthetic self-test
python3 scripts/mesh/test_glb_inspect.py
```

**Done (astra):** `trellis2.glb` with PBR textures, plus `trellis-text.glb` when `--with-text` was installed.

**Done (Mac):** `trellis2-mps.glb` at pipeline `512` with Metal baker (`_BACKEND=metal`), PNG PBR textures, `alphaMode` from alpha, and dep SHAs. Decode `Mesh.fill_holes` stays skipped. Post-bake prune position-welds coincident verts and drops a cube-floor sheet only when a thicker piece remains. Welded rematch `trellis2-mps-welded.glb` is 2 components (largest min extent is the body, not the floor), still non-watertight, PNG PBR intact. Component sizes sum to the face count. `trellis2-mps-shell.glb` is a separate watertight crust of that body (nearest-vertex UV copy, not a re-bake).

Room bake defaults: ~**100k** faces (bake `decimation_target`), texture **1024**, PNG, `alphaMode` set from the alpha channel (`MASK` / `BLEND` when not opaque). Stock upstream sample (1e6 faces, 4096 webp, `OPAQUE`) is a hero render, not a prop.

## License

| Piece | Note |
|-------|------|
| TRELLIS / TRELLIS.2 weights + code | MIT |
| Stock textured GLB bake (CUDA) | Calls **nvdiffrast** (NVIDIA Source Code License, research/evaluation only). Lab OK. Service bake must not import it. |
| Mac Metal bake (`mtldiffrast`) | Published MIT; still a nvdiffrast-API reimplementation. Lab checkout only — do not vendor into zerollama. |
| TRELLIS.2 bake replacement | Keep **cumesh** UVs + PBR volume; replace the UV-space `dr.rasterize` only. |
| Text XL color bake | Same nvdiffrast issue; the MIT artifact is the gaussian **ply**, which is not a room GLB. |
| DINOv3 (TRELLIS.2 encoder) | Commercial use allowed; redistribute weights → attach DINOv3 license + “Built with DINOv3”. Camenduru mirror is a convenience download, not a license change. |
| BiRefNet (optional rembg) | Different model from RMBG-2.0; only used when `--preprocess-image` and the still lacks alpha. Do not download CC BY-NC RMBG-2.0 for the Mac acceptance path. |

## Service shape (decided after the lab files)

Lab compare (`compare-note.json`): TRELLIS.2 GLB has **PBR** (`baseColorTexture` + `metallicRoughnessTexture`, `alphaMode=MASK`); text XL GLB is **color-only**. Rooms need metal/roughness under an HDRI, so the product paths are:

- **Local Mac:** prompt → MLX still → rembg → MPS TRELLIS.2 wrapper → GLB.
- **astra:** prompt → still → rembg → CUDA TRELLIS.2 wrapper (one free 4090, exclusive hold) → GLB.

Text XL is optional when skipping image gen is worth a color-only mesh. Do not default the HTTP route to text XL. Quality claims still need a visual pass under an environment light. Astra remains the CUDA reference bake.

## Lab text → scene plan (Mac)

`scripts/mesh/scene_plan_mac.py` is a shopping-list planner on top of the prop path. The chat model returns names, image prompts, sizes in meters, and `on` (`floor` or another object). It does not invent world coordinates: a TRELLIS bake is a unit cube, so free positions would leave every prop the same size and plant the center in the floor. This script places by support contact: uniform scale is the largest factor that keeps the mesh inside the planned AABB, objects pack along +X on the support top (cursor advances only after a successful place), bottom sits on the support’s top, and room exit / off-support / AABB overlap are refused. Writes `scene.json`, and optionally `--bake`s each prop through `mac_object_still.sh` plus the resident on `127.0.0.1:18091`. Client only — does not start serve, never binds `:11434` / `:8081`. SceneSmith, Holodeck, Hunyuan, and Ctrl-Room stay unwired.

```bash
python scripts/mesh/scene_plan_mac.py --self-test
python scripts/mesh/scene_plan_mac.py \
  --prompt "a desk with a mug on it" --plan-model llama3.2 --plan-only
# after mac_mesh_lab.sh is up:
python scripts/mesh/scene_plan_mac.py \
  --prompt "a desk with a mug on it" --plan-model llama3.2 --bake
```

## After the files exist (next slices)

1. **Service bake** without nvdiffrast — texture compare against the stock lab `trellis2.glb` (cumesh UVs + PBR volume stay; replace UV `dr.rasterize` only).
2. **`POST /v1/meshes`** on astra — short handler, exclusive GPU hold ([`server/video_exclusive.go`](../server/video_exclusive.go)), artifacts under `$OLLAMA_MODELS/generated/meshes/`. A prompt on the TRELLIS.2 tag means image gen → rembg → this wrapper, one job id. Do not clone [`server/video_generate.go`](../server/video_generate.go).
3. **Mac preview** — `<model-viewer>` with an environment image so metal/roughness show.

## Later catalog (not installed)

Pixal3D (photo fidelity, same TRELLIS.2 env), Step1X-3D (Apache, ~28 GB), PartCrafter, TripoSG / Hi3DGen (shape checks), Direct3D-S2, TripoSR (speed floor), Stable Fast 3D / SAM 3D (gated). Hunyuan3D 2.1 / Omni stay out (community license blocks EU, UK, South Korea; 3.x has no weights). Hosted Hunyuan3D 3, Tripo, Meshy, Rodin = quality ceiling only (no keys, no proxies).
