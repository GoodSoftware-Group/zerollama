# MiniMax Music 3 — findings & learnings

**Why this doc:** Capture traps so the next agent does not re-install Comfy, wait on CUDA Omni to hear a clip, or wire HTTP like Wan while missing the two hooks that make Wan jobs fetchable. Operator how-to: [music-c.md](./music-c.md).

**Date:** Aug 2026 · **Lab:** Mac aarch64 (mlx-audio 8-bit ~10 s clip) · **Gold:** [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni) `sglang_omni/models/minimax_music3/` (not cloned by default; sibling `../sglang` has **no** Music 3 files).

---

## Problem (why we looked)

Zerollama `/v1/audio/speech` is Piper / remote TTS. MiniMax **H3** AudioVAE is joint video+audio, not songs. The [cloud Music API](https://platform.minimax.io/docs/guides/music-generation) is a different product (covers, lyrics-gen). Local open weights exist (`MiniMaxAI/MiniMax-Music3`) but every “easy” path had a license, device, or disk trap.

---

## What we tried / rejected

| Path | Verdict | Why |
|------|---------|-----|
| ComfyUI `comfy/ldm/minimax_music/` | **Read-only** | GPL-3.0. Geometry is useful; **lyrics normalize is not Omni**. Dumping Comfy prompts as rematch gold would fail tokenizer/CFG later. |
| `sgl-omni serve` on this Mac | **No** | Omni acoustic path **requires CUDA**. Waiting on it meant never hearing audio. |
| HF diffusers ModularPipeline / 57 GB fp32 | **Not first** | CUDA-first packing; disk. mlx-community 8-bit (~14 GB) was enough. |
| H3 AudioVAE as Music DAV | **No** | Hop 800 vs 512; BigVGAN vs DAC; 32 ch vs 128 latent folded 2×64. Snake/conv kernels can be copied. |
| Stock Qwen3 GGUF for c0 AR | **No** | 200k vocab + music specials + audio embeddings. |
| C `--generate` before a WAV | **No** | Would debug silent DiT/DAV with no ear. mlx first (~4 min for 10 s on M-series 8-bit). |
| `{job_id}` only on `WAN_OUTPUT_PATH` | **Bug** | Music used `MUSIC3_OUTPUT_PATH`. Child wrote a literal `{job_id}.wav`; `/content` looked for the uuid path → 502. **Fix:** expand **all** string env values in `training.py`. |
| `LookPath("python3")` for run_script | **Bug** | System 3.9 has no mlx-audio. Wan already pointed at a venv. **Fix:** `.venv-music/bin/python` then `ZEROLLAMA_MUSIC_PYTHON`. |
| `max_new_tokens` always overwriting `duration` | **Bug** | Omni frames/25 is useful only when duration is omitted. Both set → honor **duration**. |
| Sync speech bytes for Music 3 | **No** | Songs are minutes of wall. 202 + poll, same as `/v1/videos`. Document the OpenAI speech footgun. |

---

## What we shipped (and why each piece)

| Piece | Why |
|-------|-----|
| `scripts/audio/music3_mlx_generate.py` + pin `784b29e` | Operators never need Comfy. Env `MUSIC3_*` matches training `run_script`. Expand `{job_id}` in-process too (belt if Go/Python disagree). |
| mlx-community 8-bit under `~/.zerollama/models/` | Hear without 57 GB. Lab WAV: `/tmp/music3_10s.wav`, **44.1 kHz** (mlx native, not Omni 32 kHz). |
| `x/music-c/tools/dump_music3_contract.py` | CPU Omni contract (prompt, chunks, `aligned_mel_length(250)=861` with **float** intermediates). Do not dump Comfy. |
| C `--info` / `--tokenize` / synthetic `--decode-audio` | Weightless geometry + Snake kernel. `--tokenize` is **prompt pack** until BMTL music vocab exists. |
| `POST /v1/audio/generations` | Clients pick a **tag**, not a runner (same rule as video-c). |
| Exclusive GPU hold | Metal music + chat llama-server on UMA is the Wan OOM class. |
| Capability `speech` + backend `music3` | Avoid a second capability zoo before we know if TTS and TTM should split. Catalog/voices UX is the cost. |

---

## Constants that bit us

- **`aligned_mel_length`:** `int(frames * 44100/24000 * 960/512)` — C must use **double** intermediates (250 → **861**, not integer-only 859).
- **Chunk:** 200 frames / hop 100; last window can be short.
- **CFG:** AR uncond overwrites `input_ids[:, 1:-2]` with `<|audio_cfg|>=151654`; acoustic uncond row is **zeros**.
- **Seed:** Omni `blake2b(person=b"minimax-ttm")`; AR `"ar"`; DiT per `chunk_idx`. mlx-audio has its own seed path — not bit-identical to Omni.
- **HF test script** `minimax_ttm_test.py` is an **HTTP client** to Omni `/v1/audio/speech`, not a local generate.

---

## Non-goals (still)

Cloud cover / lyrics-gen. Vendoring Comfy. `sgl-omni serve` on Mac. C 8B AR / 2.4B DiT GRAPH. MiniMax cloud JSON. Dual Unsloth-style music trainer.

---

## Last checked

2026-08-17 — mlx-audio 8-bit 10 s WAV on this Mac; C weightless tests green; HTTP payload + `{job_id}` expand + venv python fixed after audit (first wire missed Wan’s two hooks).
