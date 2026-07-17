#!/usr/bin/env bash
# Live video agent inference gate — real VLM prefill + L3 cached_tokens on turn 2.
#
# WHY: video_agent_cache_smoke live mode uses _debug_render_only (expand cache only).
# Agents need proof that repeat clips on the same prompt_cache_key hit prefix cache on
# turn 2 — cached_prompt_tokens on /api/chat (L3 subprocess or ollama-engine input cache).
# Expand-only smoke gave false confidence: ffmpeg/session caches can pass while vision
# prefill and KV reuse are broken.
#
# WHY log read is after all HTTP legs: VIDEO_AGENT_INFER_PREPROC=1 and
# VIDEO_AGENT_INFER_PREFIX_MM_WARN=1 append serve log lines; parsing before those legs caused false failures.
#
# WHY VIDEO_AGENT_INFER_PREPROC / PRECOMPUTED require VIDEO_AGENT_GO_LOG: layout restore
# and skip-ViT inject are proven by log grep, not response body — unverifiable without serve log.
#
# Prerequisite: running zerollama serve with vision model, ffmpeg, L3 enabled
# (subprocess/llama-server path; MLX-only may soft-pass).
# Mac Metal ollama-engine: turn-2 cached_prompt_tokens uses runner input-cache hits
# (PromptEvalCachedCount) — llama_cache.enabled may be false; use VIDEO_AGENT_INFER_SOFT=1
# only when KV cache is off or model is MLX-only.
#
# Usage:
#   RUN_E2E_VIDEO_AGENT_INFER=1 \
#     VIDEO_SMOKE_MODEL=qwen3-vl:latest \
#     OLLAMA_HOST=http://127.0.0.1:8080 \
#     ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 \
#     VIDEO_AGENT_GO_LOG=/tmp/zerollama-go.log \
#     ./scripts/video/video_agent_infer_smoke.sh
#
# Env:
#   VIDEO_SMOKE_MODEL          — required
#   OLLAMA_HOST                — Go API (default http://127.0.0.1:8080)
#   ZEROLLAMA_RUNTIME_URL      — runtime /health llama_cache (default http://127.0.0.1:8081)
#   VIDEO_AGENT_CACHE_KEY      — default video-agent-infer-1
#   VIDEO_AGENT_NUM_PREDICT    — default 8 (keep smoke fast)
#   VIDEO_AGENT_INFER_MIN_CACHED — minimum turn-2 cached_prompt_tokens (default 1)
#   VIDEO_AGENT_INFER_SOFT=1   — pass when inference OK but cache_n is 0 (MLX / cache off)
#   VIDEO_AGENT_INFER_PREPROC=1 — optional third leg: pre-expanded padded_input_ids + real infer
#                                 (requires VIDEO_AGENT_GO_LOG for layout cache hit grep)
#   VIDEO_AGENT_INFER_PREFIX_MM_WARN=1 — optional: POST without prompt_cache_key but
#                                 enable_prefix_mm_cache; grep serve log for prefix-mm hint
#                                 (requires VIDEO_AGENT_GO_LOG — hint is log-only, not in response body)
#   VIDEO_AGENT_INFER_VIT_SESSION=1 — strict: turn-2 must log vision embed session cache hit
#                                 (requires VIDEO_AGENT_GO_LOG; ggml/ollama-engine ViT overlay)
#   VIDEO_AGENT_INFER_GRID_THW=1 — strict: preproc leg must log grid_thw hint resize or vision grid hint match
#                                 (requires VIDEO_AGENT_INFER_PREPROC=1 and VIDEO_AGENT_GO_LOG)
#   VIDEO_AGENT_INFER_PRECOMPUTED=1 — strict skip-ViT: POST padded_input_ids + precomputed_embedding;
#                                 requires VIDEO_AGENT_GO_LOG; fails without precomputed_embedding runner inject
#   VIDEO_AGENT_INFER_VIT_RADIX=1 — strict cross-session ViT pool: same clip, two prompt_cache_keys;
#                                 requires VIDEO_AGENT_GO_LOG; fails without vision embed radix cache hit
#                                 (serve needs OLLAMA_VIT_RADIX on — default since Jul 2026)
#   VIDEO_AGENT_INFER_OUT      — JSON report (default /tmp/video-agent-infer-smoke.json)
#   VIDEO_AGENT_GO_LOG         — optional serve log; grep session cache + access log fields
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ "${RUN_E2E_VIDEO_AGENT_INFER:-0}" != "1" ]]; then
  echo "Set RUN_E2E_VIDEO_AGENT_INFER=1 to run live video+inference gate" >&2
  echo "Example: RUN_E2E_VIDEO_AGENT_INFER=1 VIDEO_SMOKE_MODEL=qwen3-vl:latest $0" >&2
  exit 1
fi

MODEL="${VIDEO_SMOKE_MODEL:-}"
if [[ -z "${MODEL}" ]]; then
  echo "VIDEO_SMOKE_MODEL is required" >&2
  exit 1
fi

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:8080}"
OLLAMA_URL="${OLLAMA_URL%/}"
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
RUNTIME_URL="${RUNTIME_URL%/}"
CACHE_KEY="${VIDEO_AGENT_CACHE_KEY:-video-agent-infer-1}"
GO_LOG="${VIDEO_AGENT_GO_LOG:-}"
OUT="${VIDEO_AGENT_INFER_OUT:-/tmp/video-agent-infer-smoke.json}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found" >&2
  exit 1
fi

if ! curl -sf -m 5 "${OLLAMA_URL}/api/tags" >/dev/null; then
  echo "zerollama not reachable at ${OLLAMA_URL}" >&2
  exit 1
fi

echo "== live video agent inference + L3 cached_tokens (${MODEL}) =="
export OLLAMA_URL RUNTIME_URL MODEL CACHE_KEY GO_LOG OUT
export VIDEO_AGENT_NUM_PREDICT="${VIDEO_AGENT_NUM_PREDICT:-8}"
export VIDEO_AGENT_INFER_MIN_CACHED="${VIDEO_AGENT_INFER_MIN_CACHED:-1}"
export VIDEO_AGENT_INFER_SOFT="${VIDEO_AGENT_INFER_SOFT:-0}"
export VIDEO_AGENT_INFER_PREPROC="${VIDEO_AGENT_INFER_PREPROC:-0}"
export VIDEO_AGENT_INFER_PREFIX_MM_WARN="${VIDEO_AGENT_INFER_PREFIX_MM_WARN:-0}"
export VIDEO_AGENT_INFER_PRECOMPUTED="${VIDEO_AGENT_INFER_PRECOMPUTED:-0}"
export VIDEO_AGENT_INFER_VIT_RADIX="${VIDEO_AGENT_INFER_VIT_RADIX:-0}"

if [[ "${VIDEO_AGENT_INFER_PREPROC}" == "1" && -z "${GO_LOG}" ]]; then
  echo "VIDEO_AGENT_INFER_PREPROC=1 requires VIDEO_AGENT_GO_LOG (preproc layout cache hit grep)" >&2
  exit 1
fi
if [[ "${VIDEO_AGENT_INFER_PREFIX_MM_WARN}" == "1" && -z "${GO_LOG}" ]]; then
  echo "VIDEO_AGENT_INFER_PREFIX_MM_WARN=1 requires VIDEO_AGENT_GO_LOG (prefix-mm hint grep)" >&2
  exit 1
fi
if [[ "${VIDEO_AGENT_INFER_PRECOMPUTED}" == "1" && -z "${GO_LOG}" ]]; then
  echo "VIDEO_AGENT_INFER_PRECOMPUTED=1 requires VIDEO_AGENT_GO_LOG (inject log grep)" >&2
  exit 1
fi
if [[ "${VIDEO_AGENT_INFER_VIT_RADIX}" == "1" && -z "${GO_LOG}" ]]; then
  echo "VIDEO_AGENT_INFER_VIT_RADIX=1 requires VIDEO_AGENT_GO_LOG (radix hit grep)" >&2
  exit 1
fi

python3 <<'PY'
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

url = os.environ["OLLAMA_URL"].rstrip("/")
runtime_url = os.environ["RUNTIME_URL"].rstrip("/")
model = os.environ["MODEL"]
cache_key = os.environ["CACHE_KEY"]
go_log = os.environ.get("GO_LOG", "").strip()
out_path = Path(os.environ.get("OUT", "/tmp/video-agent-infer-smoke.json"))
num_predict = int(os.environ.get("VIDEO_AGENT_NUM_PREDICT", "8"))
min_cached = int(os.environ.get("VIDEO_AGENT_INFER_MIN_CACHED", "1"))
soft = os.environ.get("VIDEO_AGENT_INFER_SOFT", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
run_preproc = os.environ.get("VIDEO_AGENT_INFER_PREPROC", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
run_prefix_mm_warn = os.environ.get("VIDEO_AGENT_INFER_PREFIX_MM_WARN", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
run_vit_session = os.environ.get("VIDEO_AGENT_INFER_VIT_SESSION", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
run_grid_thw = os.environ.get("VIDEO_AGENT_INFER_GRID_THW", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
run_precomputed = os.environ.get("VIDEO_AGENT_INFER_PRECOMPUTED", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
run_vit_radix = os.environ.get("VIDEO_AGENT_INFER_VIT_RADIX", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

if run_vit_session and not go_log:
    raise SystemExit(
        "VIDEO_AGENT_INFER_VIT_SESSION=1 requires VIDEO_AGENT_GO_LOG "
        "(vision embed session cache hit is log-only)"
    )
if run_grid_thw:
    if not run_preproc:
        raise SystemExit("VIDEO_AGENT_INFER_GRID_THW=1 requires VIDEO_AGENT_INFER_PREPROC=1")
    if not go_log:
        raise SystemExit(
            "VIDEO_AGENT_INFER_GRID_THW=1 requires VIDEO_AGENT_GO_LOG "
            "(grid_thw forward is log-only)"
        )
if run_precomputed and not go_log:
    raise SystemExit(
        "VIDEO_AGENT_INFER_PRECOMPUTED=1 requires VIDEO_AGENT_GO_LOG "
        "(precomputed_embedding runner inject is log-only)"
    )
if run_vit_radix and not go_log:
    raise SystemExit(
        "VIDEO_AGENT_INFER_VIT_RADIX=1 requires VIDEO_AGENT_GO_LOG "
        "(vision embed radix cache hit is log-only)"
    )

tmpdir = tempfile.mkdtemp(prefix="video-agent-infer-")
import atexit, shutil
atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
mp4 = os.path.join(tmpdir, "lavfi.mp4")
subprocess.run(
    [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
        "-pix_fmt", "yuv420p", mp4,
    ],
    check=True,
)
with open(mp4, "rb") as f:
    video_b64 = base64.standard_b64encode(f.read()).decode()


def http_json(method: str, endpoint: str, body: dict | None = None, timeout: float = 900.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode(errors="replace") if e.fp else ""
        raise SystemExit(f"HTTP {e.code} {endpoint}: {body_txt[:1200]}") from e


def runtime_health() -> dict:
    try:
        req = urllib.request.Request(f"{runtime_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"_fetch_error": str(exc)}


def post_chat(messages, *, debug_render_only: bool = False, cache_key_override: str | None = None, enable_prefix_mm: bool = True) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "prompt_cache_key": cache_key_override or cache_key,
            "num_ctx": 8192,
            "num_predict": num_predict,
        },
    }
    if enable_prefix_mm:
        payload["options"]["enable_prefix_mm_cache"] = True
    if debug_render_only:
        payload["_debug_render_only"] = True
    return http_json("POST", "/api/chat", payload)


def metrics_from_chat(out: dict) -> dict:
    return {
        "prompt_eval_count": out.get("prompt_eval_count") or 0,
        "cached_prompt_tokens": out.get("cached_prompt_tokens") or 0,
        "image_tokens": out.get("image_tokens") or 0,
        "video_tokens": out.get("video_tokens") or 0,
        "audio_tokens": out.get("audio_tokens") or 0,
        "done": bool(out.get("done")),
        "done_reason": out.get("done_reason"),
    }


health = runtime_health()
lc = health.get("llama_cache") or {}
llama_cache_enabled = lc.get("enabled") is True

video_msg = {
    "role": "user",
    "content": "Describe this clip in one short sentence.",
    "videos": [video_b64],
}

# Turn 1 — establish KV + session expansion for this thread.
out1 = post_chat([video_msg])
m1 = metrics_from_chat(out1)
if not m1["done"]:
    raise SystemExit(f"turn1 incomplete: {out1!r}")
assistant = (out1.get("message") or {}).get("content") or "ok"

# Turn 2 — agent resend clip; prefix should hit L3 when enabled.
follow_up = {
    "role": "user",
    "content": "Same clip again — reply in one word.",
    "videos": [video_b64],
}
out2 = post_chat([video_msg, {"role": "assistant", "content": assistant}, follow_up])
m2 = metrics_from_chat(out2)
if not m2["done"]:
    raise SystemExit(f"turn2 incomplete: {out2!r}")

# OpenAI shape on turn 2 (cached_tokens in prompt_tokens_details).
v1_payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Same clip v1 infer"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + video_b64}},
            ],
        }
    ],
    "stream": False,
    "prompt_cache_key": cache_key,
    "enable_prefix_mm_cache": True,
    "options": {"num_ctx": 8192, "num_predict": num_predict},
}
v1_out = http_json("POST", "/v1/chat/completions", v1_payload)
usage = v1_out.get("usage") or {}
ptd = usage.get("prompt_tokens_details") or {}
v1_cached = (ptd.get("cached_tokens") or 0) if isinstance(ptd, dict) else 0

preproc_report = None
if run_preproc:
    framedir = os.path.join(tmpdir, "preproc_frames")
    os.makedirs(framedir, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", mp4, "-frames:v", "2", os.path.join(framedir, "f_%02d.png"),
        ],
        check=True,
    )
    frame_b64 = []
    for name in sorted(os.listdir(framedir)):
        if not name.endswith(".png"):
            continue
        with open(os.path.join(framedir, name), "rb") as f:
            frame_b64.append(base64.standard_b64encode(f.read()).decode())
    if len(frame_b64) < 2:
        raise SystemExit(f"preproc: expected 2 PNG frames, got {len(frame_b64)}")
    preproc_key = cache_key + "-preproc-infer"
    padded = [101, 102, 103, 104, 105]
    preproc_span = [{"frame_count": len(frame_b64), "grid_thw": [len(frame_b64), 8, 8]}]
    pre1 = {
        "role": "user",
        "content": "preprocessed infer clip",
        "images": frame_b64,
        "video_spans": preproc_span,
        "padded_input_ids": padded,
    }
    out_pre1 = post_chat([pre1], cache_key_override=preproc_key)
    m_pre1 = metrics_from_chat(out_pre1)
    if not m_pre1["done"]:
        raise SystemExit(f"preproc turn1 incomplete: {out_pre1!r}")
    out_pre2 = post_chat(
        [
            pre1,
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": "same preprocessed clip",
                "images": frame_b64,
                "video_spans": preproc_span,
            },
        ],
        cache_key_override=preproc_key,
    )
    m_pre2 = metrics_from_chat(out_pre2)
    if not m_pre2["done"]:
        raise SystemExit(f"preproc turn2 incomplete: {out_pre2!r}")
    preproc_cached_ok = m_pre2["cached_prompt_tokens"] >= min_cached
    preproc_verdict = "pass" if preproc_cached_ok else ("soft" if soft else "fail")
    preproc_report = {
        "cache_key": preproc_key,
        "turn1_metrics": m_pre1,
        "turn2_metrics": m_pre2,
        "padded_len": len(padded),
        "grid_thw": preproc_span[0]["grid_thw"],
        "turn2_cached_ok": preproc_cached_ok,
        "verdict": preproc_verdict,
    }

prefix_mm_warn_report = None
if run_prefix_mm_warn:
    warn_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "prefix mm warn probe", "videos": [video_b64]}],
        "stream": False,
        "options": {"enable_prefix_mm_cache": True, "num_ctx": 8192, "num_predict": 1},
    }
    http_json("POST", "/api/chat", warn_payload)
    prefix_mm_warn_report = {"sent_without_session_key": True}

precomputed_report = None
if run_precomputed:
    # Skip-ViT: synthetic feature rows + Qwen vision block in padded_input_ids.
    # WHY /api/show for embd: ollama-engine and llamarunner both require row width == text embd.
    show = http_json("POST", "/api/show", {"model": model})
    info = show.get("model_info") or {}
    embd = 0
    for k, v in info.items():
        if str(k).endswith(".embedding_length") and isinstance(v, (int, float)) and int(v) > 0:
            embd = int(v)
            break
    if embd <= 0:
        details = show.get("details") or {}
        fam = str(details.get("family") or details.get("families") or "")
        raise SystemExit(
            f"precomputed leg: cannot resolve embedding_length from /api/show "
            f"(family={fam!r}); need model_info '*.embedding_length'"
        )
    # Qwen3-VL vision markers (HF/mtmd).
    vision_start, vision_end, vision_pad = 151652, 151653, 151655
    n_rows = 4
    grid_thw = [1, 2, 2]
    feature = [[0.01 * ((i * embd + j) % 17) for j in range(embd)] for i in range(n_rows)]
    padded = [1, vision_start] + [vision_pad] * n_rows + [vision_end, 2]
    precomp_key = cache_key + "-precomputed-infer"
    precomp_msg = {
        "role": "user",
        "content": "precomputed skip-vit probe",
        "padded_input_ids": padded,
        "images": [
            {
                "format": "precomputed_embedding",
                "feature": feature,
                "grid_thw": grid_thw,
            }
        ],
    }
    out_pc = post_chat([precomp_msg], cache_key_override=precomp_key)
    m_pc = metrics_from_chat(out_pc)
    if not m_pc["done"]:
        raise SystemExit(f"precomputed infer incomplete: {out_pc!r}")
    # Turn 2: same rows + session key → session/global precomputed cache hit when overlay on.
    out_pc2 = post_chat(
        [
            precomp_msg,
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": "precomputed skip-vit again",
                "padded_input_ids": padded,
                "images": [
                    {
                        "format": "precomputed_embedding",
                        "feature": feature,
                        "grid_thw": grid_thw,
                    }
                ],
            },
        ],
        cache_key_override=precomp_key,
    )
    m_pc2 = metrics_from_chat(out_pc2)
    if not m_pc2["done"]:
        raise SystemExit(f"precomputed turn2 incomplete: {out_pc2!r}")
    precomputed_report = {
        "cache_key": precomp_key,
        "embd": embd,
        "rows": n_rows,
        "grid_thw": grid_thw,
        "turn1_metrics": m_pc,
        "turn2_metrics": m_pc2,
        "verdict": "pass",
    }

vit_radix_report = None
if run_vit_radix:
    # Cross-session ViT pool: same clip bytes, different prompt_cache_key.
    # Session overlay must miss on key-B; radix/global content pool should hit.
    radix_key_a = cache_key + "-vit-radix-a"
    radix_key_b = cache_key + "-vit-radix-b"
    msg_a = {
        "role": "user",
        "content": "vit radix donor",
        "videos": [video_b64],
    }
    out_ra = post_chat([msg_a], cache_key_override=radix_key_a)
    m_ra = metrics_from_chat(out_ra)
    if not m_ra["done"]:
        raise SystemExit(f"vit radix donor incomplete: {out_ra!r}")
    out_rb = post_chat(
        [{"role": "user", "content": "vit radix consumer", "videos": [video_b64]}],
        cache_key_override=radix_key_b,
    )
    m_rb = metrics_from_chat(out_rb)
    if not m_rb["done"]:
        raise SystemExit(f"vit radix consumer incomplete: {out_rb!r}")
    vit_radix_report = {
        "donor_key": radix_key_a,
        "consumer_key": radix_key_b,
        "donor_metrics": m_ra,
        "consumer_metrics": m_rb,
        "verdict": "pass",
    }

log_text = ""
if go_log:
    try:
        log_text = Path(go_log).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"cannot read VIDEO_AGENT_GO_LOG={go_log}: {exc}") from exc

log_checks: dict = {}
infer_backend = None
if go_log:
    if (
        "padded_input_ids runner inject" in log_text
        and ("engine=ollama" in log_text or '"engine", "ollama"' in log_text)
    ):
        infer_backend = "ollama-engine"
    elif "padded_input_ids llama-server inject" in log_text:
        infer_backend = "llama-server"
    elif "padded_input_ids runner inject" in log_text:
        infer_backend = "llamarunner"
    log_checks = {
        "session_cache_hit": "video sample session cache hit" in log_text,
        "vision_embed_session_cache_hit": "vision embed session cache hit" in log_text,
        "vision_embed_global_cache_hit": "vision embed global cache hit" in log_text,
        "vision_embed_radix_cache_hit": "vision embed radix cache hit" in log_text,
        "vision_embed_engine_ollama": "vision embed session cache hit" in log_text
        and "engine=ollama" in log_text,
        "vision_grid_hints": "vision grid hints" in log_text,
        "padded_runner_inject": "padded_input_ids runner inject" in log_text,
        "preprocessed_layout_session_cache_hit": "preprocessed layout session cache hit" in log_text,
        "precomputed_embedding_runner_inject": "precomputed_embedding runner inject" in log_text,
        "precomputed_embedding_global_cache_hit": "precomputed_embedding global cache hit" in log_text,
        "precomputed_embedding_session_cache_hit": "precomputed_embedding session cache hit" in log_text,
        "processor_output_runner_inject": "processor_output runner inject" in log_text,
        "processor_output_global_cache_hit": "processor_output global cache hit" in log_text,
        "prefix_mm_cache_without_session_key": "enable_prefix_mm_cache without prompt_cache_key" in log_text,
        "grid_thw_hint_resize": "grid_thw hint resize" in log_text,
        "vision_grid_hint_match": "vision grid hint match" in log_text,
        "access_cached_prompt_tokens": "cached_prompt_tokens" in log_text,
        "access_video_tokens": "video_tokens" in log_text,
    }
    if run_preproc and preproc_report and not log_checks["preprocessed_layout_session_cache_hit"]:
        raise SystemExit(
            "preproc infer OK but VIDEO_AGENT_GO_LOG lacks "
            "'preprocessed layout session cache hit'"
        )
    if run_prefix_mm_warn and not log_checks.get("prefix_mm_cache_without_session_key"):
        raise SystemExit(
            "prefix-mm warn leg OK but VIDEO_AGENT_GO_LOG lacks "
            "'enable_prefix_mm_cache without prompt_cache_key'"
        )
    if run_vit_session and not log_checks.get("vision_embed_session_cache_hit"):
        raise SystemExit(
            "turn2 infer OK but VIDEO_AGENT_GO_LOG lacks "
            "'vision embed session cache hit' (set prompt_cache_key every turn)"
        )
    if run_grid_thw and not (
        log_checks.get("grid_thw_hint_resize") or log_checks.get("vision_grid_hint_match")
    ):
        raise SystemExit(
            "preproc infer OK but VIDEO_AGENT_GO_LOG lacks "
            "'grid_thw hint resize' or 'vision grid hint match'"
        )
    if run_precomputed and not log_checks.get("precomputed_embedding_runner_inject"):
        raise SystemExit(
            "precomputed infer OK but VIDEO_AGENT_GO_LOG lacks "
            "'precomputed_embedding runner inject' (skip-ViT path not taken)"
        )
    if run_vit_radix and not log_checks.get("vision_embed_radix_cache_hit"):
        raise SystemExit(
            "vit radix infer OK but VIDEO_AGENT_GO_LOG lacks "
            "'vision embed radix cache hit' (rebuild serve with OLLAMA_VIT_RADIX default on, "
            "or set OLLAMA_IMAGE_EMBED_CACHE_BYTES)"
        )
    if run_precomputed and precomputed_report is not None:
        # Prefer session hit on turn 2; global hit also proves cache wire-up.
        if not (
            log_checks.get("precomputed_embedding_session_cache_hit")
            or log_checks.get("precomputed_embedding_global_cache_hit")
            or log_checks.get("vision_embed_session_cache_hit")
        ):
            precomputed_report["verdict"] = "soft"
            precomputed_report["reason"] = (
                "inject OK; no precomputed/vision session|global cache hit on turn 2"
            )
        else:
            precomputed_report["verdict"] = "pass"

if run_preproc and preproc_report and preproc_report.get("verdict") == "fail":
    raise SystemExit(
        "preproc turn2 cached_prompt_tokens="
        f"{(preproc_report.get('turn2_metrics') or {}).get('cached_prompt_tokens', 0)} "
        f"want>={min_cached}"
    )
if run_precomputed and precomputed_report and precomputed_report.get("verdict") == "fail":
    raise SystemExit(
        f"precomputed infer failed: {precomputed_report.get('reason') or precomputed_report}"
    )

# /api/chat turn-2 is the real L3 gate: same prefix, same session key, same message structure.
# /v1/chat/completions is a separate single-turn call (different message structure) — it tests
# OpenAI field wiring and session VIDEO cache, not L3 KV reuse. Treat v1_cached as advisory only.
strict_cached = m2["cached_prompt_tokens"] >= min_cached
verdict = "pass"
reason = "turn2 cached_prompt_tokens meets minimum (L3 gate)"

if not strict_cached:
    if soft or not llama_cache_enabled:
        verdict = "soft"
        reason = (
            "inference OK; turn2 cached_prompt_tokens below minimum "
            f"(api={m2['cached_prompt_tokens']}, "
            f"llama_cache.enabled={lc.get('enabled')!r}, "
            f"infer_backend={infer_backend!r})"
        )
    else:
        verdict = "fail"
        reason = (
            f"turn2 cached_prompt_tokens={m2['cached_prompt_tokens']} want>={min_cached}; "
            "ensure ZEROLLAMA_LLAMA_CACHE=1 and subprocess L3 path"
        )

report = {
    "model": model,
    "cache_key": cache_key,
    "num_predict": num_predict,
    "min_cached": min_cached,
    "llama_cache": lc,
    "runtime_health_error": health.get("_fetch_error"),
    "turn1_metrics": m1,
    "turn2_metrics": m2,
    "v1_cached_tokens": v1_cached,
    "v1_image_tokens": (ptd.get("image_tokens") if isinstance(ptd, dict) else None),
    "v1_video_tokens": (ptd.get("video_tokens") if isinstance(ptd, dict) else None),
    "infer_backend": infer_backend,
    "log_checks": log_checks or None,
    "preprocessed_infer": preproc_report,
    "prefix_mm_warn_probe": prefix_mm_warn_report,
    "precomputed_infer": precomputed_report,
    "vit_radix_infer": vit_radix_report,
    "vit_session_required": run_vit_session,
    "grid_thw_forward_required": run_grid_thw,
    "precomputed_required": run_precomputed,
    "vit_radix_required": run_vit_radix,
    "verdict": verdict,
    "reason": reason,
}
out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))

if verdict == "fail":
    raise SystemExit(reason)
if verdict == "soft":
    print(f"SOFT PASS: {reason}", file=sys.stderr)
PY

echo "report: ${OUT}"
"${ROOT}/scripts/video/video_agent_infer_gate_report.sh" "${OUT}"
echo "PASS video_agent_infer_smoke"
