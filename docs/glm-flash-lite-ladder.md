# GLM Flash Lite ladder (GF0–GF7)

**Upstream:** [sybil-solutions/glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) — EXL3 GLM-5.3-Flash on one 24 GB card via VRAM expert cache + exclusive RAM tier + NVMe store + AVX2 CPU lane.  
**Target host:** **astra** (dual RTX 4090 + `/mnt/ssd2`) — cozmic remains an alternate if we move the sidecar later.  
**Related in-tree:** [flash-moe.md](./flash-moe.md) (M16), [freetoken-moe-lab.md](./freetoken-moe-lab.md) (same FreeToken ideas they credit).

**Status (Oct 2026):** **GF0–GF5 done; GF6 parked; GF7 standing.** Sidecar `:30000` (`GLM53_NV_RAM_GB=55` → ~6257 RAM slots); tag `glm-5.3-flash` (`openai-remote`) proxies `/v1` + `/api/chat`. Next only if anemll grows a CPU-miss flag (**GF6**).

---

## Why this ladder exists

| Temptation | Why we do not |
|------------|----------------|
| Vendor their CUDA/`nv2` kernels into llama.cpp | Different quant (EXL3), different engine (exllamav3); blows the pin |
| Pretend Flash-MoE already runs GLM-5.3-Flash EXL3 | Flash-MoE is **GGUF + anemll sidecar**; this checkpoint is EXL3 |
| Force the stack onto Mac UMA first | Their recipes are discrete PCIe + AVX2 + local NVMe `O_DIRECT` |
| Bind **11434 / 8081** to “verify” | Production ports — lab sidecar only (e.g. `:30000`) |

**Product shape:** same as remote-tts / SGLang — **sidecar OpenAI `/v1`**, then optional Go proxy tag. Steal **policy ideas** into FreeToken/Flash-MoE; do **not** reimplement their host engine.

---

## Capability map (what we already have)

| Their piece | Zerollama today | Ladder use |
|-------------|-----------------|------------|
| OpenAI `/v1` | Native chat + Hermes surfaces | GF2–GF3 expose their server through us |
| Hot expert cache / SSD experts | Flash-MoE slot-bank + sidecar | GF5 rematch policies; not their CLOCK cache |
| Layer-ahead prefetch | `--moe-prefetch-temporal` | Already wired on Flash-MoE |
| CPU experts vs PCIe fill | FreeToken \(q^\star\) **sim + doctor** | GF4 measure; GF6 only if anemll grows a flag |
| Host / VRAM admission | Phase 8–11 broker + `check_gguf_host_budget` | GF3 pin so sidecar does not thrash GGUF |
| GLM family | GLM-4.x / `glm4moelite` / OCR | **Not** 5.3-Flash EXL3 |
| Exclusive RAM + NVMe `O_DIRECT` store | — | Stay in their exllamav3 sidecar |

---

## Rungs

| Rung | Goal | Owner | Exit | Status |
|------|------|-------|------|--------|
| **GF0** | **Host inventory** | Ops | GPU name + free VRAM; free RAM; NVMe fs (xfs/ext4), free ≥ 250 GB local; AVX2; logical CPUs ≥ 40 preferred | **Done (astra)** |
| **GF1** | **Engine smoke (no zerollama)** | Ops | Weights + expert store + OpenAI `/v1` answering on a lab port; decode usable (≥ ~15 tok/s). Packaging = whatever you prefer (venv/systemd **or** their image) | **Done (astra)** |
| **GF2** | **Direct clients** | Ops | Hermes / OpenAI SDK → `http://127.0.0.1:30000/v1` (loopback); model id `glm-5.3-flash`; document auth | **Done (astra loopback)** |
| **GF3** | **Zerollama proxy tag** | Go | Manifest `modality_backends.inference=openai-remote` → OpenAI-compatible base URL; lab port only; no GGUF load | **Done (astra)** |
| **GF4** | **FreeToken rematch on astra** | Lab | Capture `/stats` hit mix; compare to `AdviseProfile("4090")` / README C1; write findings | **Done (astra)** |
| **GF5** | **Borrow into Flash-MoE docs only** | Docs | Document exclusive-tier + CPU-lane ideas as **watch** against anemll; no code unless fork exposes flags | **Done** |
| **GF6** | **Optional anemll CPU-miss split** | Fork watch | Only if anemll (or our Flash-MoE binary) gains a real CPU-expert / pin-budget path; wire env + doctor | Parked |
| **GF7** | **Do not vendor EXL3** | Policy | Explicit non-goal: no `exllamav3` / `nv2` in-tree; keep Docker digest pin in this doc | Standing rule |

**Do not** jump GF0 → GF3 without GF1 numbers. A proxy that 503s or starves TTS/GGUF on the same GPU is worse than a direct `:30000` URL.

---

## GF0 — inventory

```bash
# GPU host, read-only (needs /dev/nvidia* — not sandboxed)
./scripts/phase/gf0_cozmic_inventory.sh
```

### astra result (2026-10-09)

| Item | Value | Verdict |
|------|-------|---------|
| GPUs | 2× RTX 4090 24 GB (AD102), driver 595.99.02 | OK (≥23 GiB) |
| GPU0 free / link | ~14.6 GiB free; **PCIe Gen4 x16** (max 4 / width 16) | Prefer for GF1; idle ASPM can *report* Gen1 — recheck under load |
| GPU1 free / link | ~24.0 GiB free; **max Gen3, stuck x1** (width downgraded) | **Do not use** for MoE offload (x1 starves PCIe) |
| Occupants | kwin/brave on 0; `.venv/bin/python` ~7.9 GiB on 0 | Ask before killing; GF1 should use idle GPU or unload python |
| RAM | 125 GiB total, ~106 GiB MemAvailable | **`nvme` @ 55 g** (not `fast` — needs ~238 GiB free) |
| CPU | i9-13900K, 32 logical; AVX2+FMA+F16C | OK; below their “40 logical” measured layout → derived pin layout |
| Disk | `/mnt/ssd2` = **CT2000BX500 SATA** ext4 (~1.2 TiB free) | Weights OK; not true NVMe |
| Real NVMe | `nvme0n1` Samsung **9100 PRO 2 TB** — **ext4** `/mnt/nvme` (operator-approved `mkfs`) | Expert store `/mnt/nvme/glm53` |
| Docker | absent | Prefer venv/systemd |

| Recipe (upstream) | RAM | Storage | astra |
|-------------------|-----|---------|-------|
| `nvme` 55 GiB (accepted) | cap ~55 GiB | ~243 GB | **Chosen** |
| `nvme` 16 GiB | cap ~16 GiB | same store | fallback if RAM contended |
| `fast` all-RAM | ~238 GiB free | 125 GB weights | No — only 125 GiB box |

Record further runs under **Findings** below.

---

## GF1 — engine smoke (what it actually is)

**Exit:** on **astra**, something that is **not** zerollama serves OpenAI `/v1` for `glm-5.3-flash`, loads once, and answers a short chat at a usable decode rate. Prefer **`CUDA_VISIBLE_DEVICES=0`** (GPU1 link width stuck at **x1**). No Go proxy yet (that is GF3).

GF1 is three artifacts + one process:

| Artifact | Size | Role |
|----------|------|------|
| EXL3 weights (`turboderp/GLM-5.3-Flash-exl3` @ pinned rev) | ~125 GB | Non-experts + checkpoint; experts are *not* all kept in VRAM |
| Expert store from `pack-store` | ~117 GB | One 4K-aligned record per expert on **local** NVMe (`O_DIRECT`; xfs/ext4) |
| Runtime | — | exllamav3 + their `nv2` host/CUDA kernels + OpenAI server |

```text
  download weights  →  pack-store (once)  →  start server (nvme | fast | …)
                              ↓
                    http://127.0.0.1:PORT/v1/models
                    http://127.0.0.1:PORT/v1/chat/completions
```

**Pass criteria (practical):** `/v1/models` lists the model; one uncapped natural-length chat finishes; note tok/s. Upstream’s registry speed gate is ~15 tok/s — use that as a soft bar, not a zerollama CI gate.

**Why a RAM *cap* (55 GiB) matters for `nvme`:** the exclusive RAM tier is sized from the process cgroup max. Without a cap, the page cache / swap can defeat the tier math. Docker `--memory` is how *they* set that cgroup; on bare metal use **systemd `MemoryMax=`** (or cgroupv2) the same way. `MemoryMax` should match swap accounting so the process cannot escape via swap.

**Docker is optional packaging.** Upstream’s README leads with a GHCR image because it pins CUDA/exllamav3/`nv2` and makes the 55 GiB cgroup easy. Preferable for us: clone [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite), follow their `skills/glm53-offload-setup/SKILL.md`, run from a venv/systemd unit on cozmic. Same exit criteria. Image digests stay useful as a *known-good* reference if a native rebuild fails (esp. Blackwell sm_120 — their binary is measured on sm_86/89).

**Pins (engine, not container):**

| Item | Value |
|------|--------|
| Weights | `turboderp/GLM-5.3-Flash-exl3` rev `332ab457b709b7ba30dd9a448be5de03b80a7ac9` |
| Mode | `nvme` if ≥ ~60 GiB free + NVMe; `fast` only if ~238 GiB free |
| Lab port | e.g. `30000` — never production `11434` / `8081` / `2083` |

**Coexistence:** unload TTS/GGUF on that GPU first, or pick another device. Zerollama’s VRAM broker does **not** see this process until GF3 marks a tag external.

---

## GF2 — clients without proxy

Point OpenAI-compatible clients at the sidecar (astra lab; bind is loopback):

```text
base_url = http://127.0.0.1:30000/v1
model    = glm-5.3-flash
auth     = unused locally (pass any string / "none")
```

```bash
# curl
curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}}'

# OpenAI Python SDK (venv has openai)
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="none")
c.chat.completions.create(model="glm-5.3-flash", messages=[{"role":"user","content":"hi"}])
```

**Restart sidecar:** `./scripts/phase/gf1_serve_nvme_astra.sh` (logs → `/mnt/ssd2/glm53-serve.log`).  
**Proxy smoke (lab):** `./scripts/phase/gf_smoke_openai_remote.sh` (starts `:11436` if needed; never `:11434`).  
Optional later: Caddy path / LAN bind. Do **not** put this on `:2083` without an explicit operator choice.

---

## GF3 — zerollama proxy tag

**Why:** one catalog for Hermes (`/api/tags` + `/v1/chat/completions`) without loading EXL3 into llama-server.

| Piece | Choice |
|-------|--------|
| Manifest | `modality_backends.inference: openai-remote` + `backend_paths.openai_url` / `openai_model` |
| Env fleet default | `ZEROLLAMA_OPENAI_REMOTE_URL` (overridden by `backend_paths.openai_url`) |
| Wire | `server/openai_remote_chat_proxy.go` — `/v1/chat/completions` + `/api/chat` (text; non-stream upstream) |
| Register | `./scripts/phase/gf3_register_openai_remote.sh` → config-only tag (no GGUF) |
| Non-goals | Tools/vision fallback into ggml for this tag; EXL3 inside runtime |

```bash
./scripts/phase/gf3_register_openai_remote.sh
./scripts/phase/gf_smoke_openai_remote.sh   # lab :11436 → sidecar :30000
# production :11434 / :11435 need a rebuild+restart of *that* binary to pick up openai-remote
```

---

## GF4 — FreeToken rematch (astra, measured)

Sidecar `GET /stats` after a warm decode (`nvme`, exclusive RAM, CPU tier on). FreeToken Table-1 advice via `go run ./x/freetokenlab/cmd/sim` profile **`4090`**.

| Metric | astra (this run) | Upstream README C1 (3090, 55 GiB) | FreeToken `4090` lab |
|--------|------------------|----------------------------------|----------------------|
| Hit mix / tok | After `GLM53_NV_RAM_GB=55`: VRAM **146** / RAM **172** / NVMe **19** (43 / 51 / 6 %) | ~120 / 175 / 23 (37.7 / 55 / 7.2 %) | — |
| CPU experts / tok | ~**150** (`cpu_busy_ms` ≈ 96 @ 55 GiB) | CPU lane ~18 % of step time | \(q^\star\): **2 fill + 2 CPU** of 4 misses |
| Layer-ahead | `la_used` ~99 % | claimed −NVMe misses | Flash-MoE `--moe-prefetch-temporal` cousin |
| Decode | ~**7.3 tok/s** wall @ 55 GiB RAM tier (was ~5.4 @ 42.5 GiB) | **19.5 tok/s** registry | — |
| Tier sizes | VRAM cache 1216 / 11.5 GB; RAM **6257** / **55.0 GiB** exclusive | ~1300 VRAM; ~4800 RAM @ 55 GiB cgroup | slot-bank advice is GGUF-shaped, not EXL3 |

**Read-out:** with `GLM53_NV_RAM_GB=55` (launcher default), hit mix tracks their C1 recipe. FreeToken’s **CPU miss-split on 4090 is directionally correct**. Remaining gap to 19 tok/s is CPU-lane / admit copy, not PCIe gen (GPU0 is Gen4×16 under load). Optional harder cap: `systemd-run … MemoryMax=55G`.

**Do not:** invent anemll flags from this; EXL3 ≠ GGUF slot-bank.

## GF5–GF6 — policy borrow (not their engine)

1. **GF5 (done):** [freetoken-moe-lab.md](./freetoken-moe-lab.md) + [flash-moe.md](./flash-moe.md) “glm-flash-lite rematch” watch items — exclusive RAM, victim ring, layer-ahead, CPU lane.
2. **GF6:** Parked until anemll exposes CPU-expert or pin-budget knobs we can pass through `appendFlashMoEArgs()`.

---

## Non-goals (GF7)

- Vendoring `kernels/nv2`, exllamav3, or EXL3 packs into this repo
- Claiming GGUF Flash-MoE parity with their 19–28 tok/s 3090 numbers
- Running the Docker image on the **PVE hypervisor** or competing with production `:11434` / `:8081` / `:2083` without an operator ask
- Arc Pro B70 companion tier (upstream experimental; skip)

---

## Findings

| Date | Host | Rung | Note |
|------|------|------|------|
| 2026-10-09 | astra | GF0 | **PASS** — 2×4090; mode `nvme`@55g; use **GPU0**. First sandboxed `nvidia-smi` lied (no `/dev/nvidia*`); real devices present. Idle ASPM initially reported Gen1; under load GPU0 is **Gen4 x16**. GPU1 max Gen3 and **width stuck x1** — skip for MoE. |
| 2026-10-09 | astra | GF1 paths | Weights → `/mnt/ssd2/models/glm53` (SATA BX500). Expert store should prefer blank **Samsung 9100 PRO** `nvme0n1` once formatted; interim `/mnt/ssd2/glm53-nvx` if operator declines `mkfs`. |
| 2026-10-09 | astra | GF1 | **PASS** — `mkfs.ext4` `/dev/nvme0n1p1` → `/mnt/nvme` (fstab UUID `ba90c06c-…`); pack+verify store 117.3 GB @ 3.52 GB/s O_DIRECT; native venv + exllamav3 1.5.1; serve `:30000` loaded 48.9 s; RAM tier 4835 / VRAM cache 1216 slots; startup verify 0 bad. Stopped irodori TTS (~8 GiB) to load. Launcher: `serve_nvme_astra.sh`. |
| 2026-10-09 | astra | GF1 smoke | **PASS** — `/v1/chat/completions` `finish_reason=stop`; usage 18→218 tok (~5.7 tok/s wall over 38.3 s first reply — soft bar ~15 tok/s; not a PCIe-gen issue — GPU0 is Gen4×16 under load; suspect cold expert cache / first-token). Coherent jet-engine answer. |
| 2026-10-09 | astra | GF2 | **PASS (loopback)** — urllib/OpenAI-compat client → `127.0.0.1:30000/v1`; models list + short chat (`Hello from Astra!`, finish stop). No LAN/Caddy yet; auth unused. |
| 2026-10-09 | astra | GF3 | **PASS** — `openai-remote` proxy + config-only `glm-5.3-flash`; lab `:11436` → sidecar `:30000`; chat `finish_reason=stop` (“Hello there, nice to meet you!”). GPU1 still chipset ×1 (hardware later). |
| 2026-10-09 | astra | GF4 | **PASS** — `/stats` after warm decode: VRAM/RAM/NVMe **43.7/45.9/10.4 %** (~147/154/35 per tok); CPU experts ~146/tok; ~5.4 tok/s. FreeToken `4090` \(q^\star\) 2+2 matches CPU-lane weight. See GF4 table above. |
| 2026-10-09 | astra | GF5 | **PASS (docs)** — rematch watch items in freetoken-moe-lab + flash-moe. |
| 2026-10-09 | astra | polish | `GLM53_NV_RAM_GB=55` → 6257 RAM slots; decode **7.3 tok/s**; NVMe hit share 10 %→6 %. `/api/chat` openai-remote proxy smoked (`done` + “Hello there, how are you?”). |
| 2026-10-09 | astra | polish | In-repo `gf1_serve_nvme_astra.sh` + `gf_smoke_openai_remote.sh`; `transformers` in venv (HF tokenizer loads when venv python is used); rebuilt `./zerollama`. Unset stray `GLM53_NV_RAM_GB` to keep 55 GiB default. Prod `:11434`/`:11435` need operator restart for proxy. |
| 2026-10-09 | cozmic.space | — | Not required for GF0–GF5; SSH from astra still denied |

---

## Links

- Upstream how-it-works: [glm-flash-lite/docs/how-it-works.md](https://github.com/sybil-solutions/glm-flash-lite/blob/main/docs/how-it-works.md)
- FreeToken paper: [arXiv:2608.16157](https://arxiv.org/abs/2608.16157)
- ROADMAP: [M16 Flash-MoE](./ROADMAP.md) + **M28** (this ladder)
