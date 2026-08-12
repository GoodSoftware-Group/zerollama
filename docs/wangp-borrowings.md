# WanGP / mmgp borrowings (video VRAM)

**Why this doc:** Wan TI2V-5B OOMs on ~16 GB when stock Wan does a full DiT `model.to(cuda)` under `--offload_model`. [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) solves that with **[`mmgp`](https://github.com/deepbeepmeep/mmgp)** (layer/budget offload). We borrow **mmgp attach + pipe layout**, not Gradio or WanGP’s multi-model zoo.

**Related:** [wan-t2v.md](./wan-t2v.md), [ROADMAP — Video generation](./ROADMAP.md#video-generation--wan-t2v-v1-shipped), sibling map [upstream-siblings.md](./upstream-siblings.md).

**Local tree:** `/root/Wan2GP` on CT 1564; Mac lab `../Wan2GP` under `~/Sites/inference/` when present. **As-needed watch** (not weekly).

**Last checked:** 2026-08-11 — tip `7e45fe7e2110` on `main` (`mmgp==3.7.12`).

**Long-term:** Python + mmgp is a **VRAM bridge**. Pure-C home: [`dit_pager`](./dit-pager.md) + [`cuda-uma-toolkit.md`](./cuda-uma-toolkit.md) (wan-c multi-backend) — ship C for **wins or unlocks**, not a literal mmgp port.

---

## Taken

| WanGP / mmgp concept | Zerollama | WHY |
|----------------------|-----------|-----|
| Early `import mmgp` (safetensors redirect) | `wan_generate_entry.py` when `WAN_MMGP=1` | mmgp README: first import so loads go through its path |
| Pipe dict with nested `.model` | `wan_memory_hooks.patch_mmgp_profile` | Wan2GP `wan_handler.load_model`: `transformer` / `vae.model` / `text_encoder.model` |
| `offload.profile(profile_no=…)` | Same hook; default **5** on 16g TI2V | Profile 4 wants ≥32 GiB host RAM; this CT has ~24 GiB → profile 5 floor |
| Neutralize full DiT `.to(cuda)` / `.cpu()` | Patch after attach under `WAN_MMGP` | Stock Wan `textimage2video` reloads ~10 GiB and defeats budgets |
| Exclude VAE from mmgp pipe when `WAN_VAE_CPU` | `attach_mmgp_profile` | CPU encode/decode owns VAE; mmgp on VAE fights scale/latent devices |
| fp32 cast on time-mod `e` | `patch_mmgp_fp32_time_mod` | Stock asserts `e.dtype == float32`; mmgp + `convert_model_dtype` can yield bf16 |
| Pin `mmgp==3.7.12` | `install_wan_video.sh` | Match Wan2GP `requirements.txt` / PyPI |

---

## Deferred (explicit non-goals this pass)

| Feature | WHY not now |
|---------|-------------|
| `wgp.py` Gradio UI | Control plane is `/v1/videos` + media uploads |
| WanGP `WanAny2V` / `fast_load_transformers_model` | Stock `generate.py` + post-init `offload.profile` first; borrow loader only if attach fails |
| Multi-model zoo (LTX, MiniMax H3, …) | ROADMAP **v1.4+** (**LTX then H3**). **LTXV distilled first slice** shipped as `backend=ltx` — [ltx-t2v.md](./ltx-t2v.md); H3 later — [h3-cuda-port.md](./h3-cuda-port.md) |
| SageAttention / TeaCache | Speed only; we already force SDPA for SM120 |
| Pure-C diffusion + paging | **In progress (wins/unlocks):** [dit-pager.md](./dit-pager.md), [cuda-uma-toolkit.md](./cuda-uma-toolkit.md); Python mmgp remains the product bridge |

---

## Env reference

| Variable | Default (16g TI2V) | Role |
|----------|-------------------|------|
| `WAN_MMGP` / `ZEROLLAMA_WAN_MMGP` | `1` | Enable mmgp wrap |
| `WAN_MMGP_PROFILE` / `ZEROLLAMA_WAN_MMGP_PROFILE` | `5` | mmgp profile 1–5 (5 = VerylowRAM_LowVRAM) |
| `WAN_MMGP_QUANTIZE` / `ZEROLLAMA_WAN_MMGP_QUANTIZE` | `0` | `1` → `quantizeTransformer=True` (int8) if still OOM |

---

## Code map

| Area | Path |
|------|------|
| Attach + neutralize | `scripts/video/wan_memory_hooks.py` |
| Early import | `scripts/video/wan_generate_entry.py` |
| Job env | `server/video_generate.go`, `scripts/video/wan_video_generate.py` |
| Install pin | `scripts/video/install_wan_video.sh` |
