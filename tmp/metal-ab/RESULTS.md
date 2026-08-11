# Metal Lab A A/B — M4 Max (Jul 2026)

**Host:** Apple M4 Max, ~128 GB unified memory  
**Model:** Llama-3.2-3B Instruct Q4_K_M (ollama `llama3.2:3b` blob)  
**Method:** `llama-bench -ngl 99 -fa 1 -p 512,2048 -n 128 -r 3`  
**Binaries:** vendor `86d86ed4` (+ patch 0093); RotorQuant `../llama-cpp-rotorquant` @ `08e025c` Metal  

**Trust:** **v2** (2026-07-25 ~22:04) after GPU idle — use these. Earlier `*.noisy.log` run was contended (`bmtl-bench` / `zerollama bench`); discard.

## Results v2 (tok/s)

| Leg | cache | pp512 | pp2048 | tg128 | vs f16 tg |
|-----|-------|------:|-------:|------:|----------:|
| **stock_f16** | f16/f16 | **2115** | **1997** | **151** | 1.00× |
| stock_q8 | q8_0/q8_0 | 1850 | 1849 | **137** | 0.91× |
| tbq | tbq4_0/tbq3_0 | 1521 | 1355 | 32 | 0.21× |
| turbo3 (rotor) | turbo3/turbo3 | 1189 | 942 | 43 | 0.28× |
| planar3_f16 | planar3/f16 | 1087 | 865 | 64 | 0.42× |
| planar3 | planar3/planar3 | 380 | 125 | 43 | 0.29× |
| iso3 | iso3/iso3 | 353 | 100 | 29 | 0.19× |
| qjl (pre-0097) | qjl1_256/q4_polar | — | — | — | **Abort** Metal SET_ROWS |
| f16_tbq3 (D0) | f16/tbq3_0 | 1645 | 1494 | 45 | 0.30× — V-only still slow |
| f16_tbq4 (D0) | f16/tbq4_0 | 1653 | 1436 | 43 | 0.29× |
| q8_tbq3 (D0) | q8_0/tbq3_0 | 1179 | 512 | 34 | 0.22× |

Logs: `tmp/metal-ab/v2/*.log` · JSON: `tmp/metal-ab/l2-rotorquant-metal-ab.json`

## Lab D quiet v3 (after 0097+0098, r=3)

| Leg | cache | pp512 | pp2048 | tg128 | vs f16 tg |
|-----|-------|------:|-------:|------:|----------:|
| stock_f16 (recheck) | f16/f16 | 2066 | 1952 | **145** | 1.00× |
| f16_polar | f16/q4_polar | 1709 | 1455 | 36 | 0.25× |
| **qjl_polar** | qjl1_256/q4_polar | 934 | **350** | **37** | 0.26× |

Logs: `tmp/metal-ab/d1b/*.v3.log`

## Related: Lab Q (Qwen2.5 TQ + MLX)

See [`tmp/qwen25-tq/RESULTS.md`](../qwen25-tq/RESULTS.md) — andrei-ace adaptive `-ctk tqk` on Qwen2.5-1.5B (VRAM/quality win, 0.40× f16 tg); MLX A/B does not beat Metal here.

## Verdict (Metal)

**No-merge planar/iso** — confirmed on quiet GPU.

- Prefill collapse is real: planar3/iso3 pp2048 ~0.05–0.06× stock f16.
- Best compressed rotor leg here is **planar3_f16** (0.42× tg) — still loses badly to stock.
- Stock on Mac 3B: prefer **f16**; q8_0 is close on decode (0.91×) but not a win like 5080’s 8B q8 story.
- TBQ / turbo3 / asymmetric TBQ-V: VRAM opt-in only (tok/s FAIL merge). TheTom “V is free” ≠ Metal decode free.
- QJL/Polar `speed` (**0097+0098**): runs; tg ~0.26× f16; pp2048 ~0.18× f16 — **tok/s FAIL merge**. VRAM-only experimental.

## Noisy vs quiet (why re-ran)

| Leg | noisy tg128 | quiet tg128 |
|-----|------------:|------------:|
| f16 | 114 | **151** |
| q8_0 | 57 | **137** |
| tbq | 25 | 32 |
| planar3 | 23 | 43 |
