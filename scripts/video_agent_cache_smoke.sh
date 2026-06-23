#!/usr/bin/env bash
# Video agent cache smoke — unit gate + agent-turn modality test + optional live E2E.
#
# WHY: SGLang xfer Tier 1 caches must survive the full agent loop (same clip + session key),
# not only modality unit tests. Modality test models turn-2 resend; live E2E proves ffmpeg + logs.
#
# Usage:
#   ./scripts/video_agent_cache_smoke.sh
#
# Live E2E (vision model + ffmpeg + running serve):
#   RUN_E2E_VIDEO_AGENT=1 \
#     VIDEO_SMOKE_MODEL=qwen3-vl:latest \
#     OLLAMA_HOST=http://127.0.0.1:8080 \
#     VIDEO_AGENT_GO_LOG=/tmp/zerollama-go.log \
#     ./scripts/video_agent_cache_smoke.sh
#
# Full inference + L3 cached_tokens (separate script):
#   RUN_E2E_VIDEO_AGENT_INFER=1 VIDEO_SMOKE_MODEL=... ./scripts/video_agent_infer_smoke.sh
#
# Live E2E also runs pre-expanded turn 2 (images + video_spans layout restore) when RUN_E2E_VIDEO_AGENT=1.
#
# Env (live):
#   VIDEO_SMOKE_MODEL     — pulled model with vision+video (required for E2E)
#   VIDEO_AGENT_CACHE_KEY — default video-agent-smoke-1
#   VIDEO_AGENT_GO_LOG    — grep for "video sample session cache hit" after turn 2
#   OLLAMA_HOST           — default http://127.0.0.1:8080
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "== video expand unit gate =="
"${ROOT}/scripts/video_expand_cache_smoke.sh"

echo "== agent thread session cache (modality) =="
go test ./server/modality/... -count=1 -run 'AgentSecondTurn|Session|PreprocessedLayout|preservesPadded|LatestUserPadded|BuildPadded|toolTurn' -short

echo "== OpenAI video_url session cache =="
go test ./openai/... -count=1 -run 'OpenAIVideoAgent' -short

echo "== Qwen3-VL + Gemma4 video span render + padded layout consume =="
go test ./model/renderers/... -count=1 -run 'Qwen3VLRenderer_videoSpans|Qwen3VLRenderer_skips|Gemma4Renderer_skips|Gemma4Renderer_video' -short
go test ./server/modality/... -count=1 -run 'PaddedLayoutConsume|BuildPadded|ollamaEngineVL|Gemma4|Mllama|Gemma3|Llama4|Lfm2|Glmocr|Mistral3|Deepseek|qwen25vl|GridTHW' -short
go test ./server/... -count=1 -run 'GgmlPadded' -short
go test ./runner/llamarunner/... -count=1 -run 'InputsFromQwen3VL|InputsFromGemma4|TestSessionEmbed' -short
go test ./runner/ollamarunner/... -count=1 -run 'Mllama|Gemma3|Llama4|Lfm2|Glmocr|Mistral3|Deepseek|Padded|GridTHW|VisionEmbed|VisionTokens' -short
go test ./llm/... -count=1 -run 'Gemma4|PaddedMultimodal|BuildLlamaServerGemma4' -short

echo "== runtime prefill cancel (wheel + disconnect) =="
python3 -m pytest runtime/tests/test_disconnect_stream.py runtime/tests/test_llama_server_http.py runtime/tests/test_llama_cpp_python.py::test_llama_cpp_python_non_stream_honors_prefill_cancel -q

if [[ "${RUN_E2E_VIDEO_AGENT:-0}" != "1" ]]; then
  echo "PASS video_agent_cache_smoke (unit+integration)"
  echo "Live E2E: RUN_E2E_VIDEO_AGENT=1 VIDEO_SMOKE_MODEL=<vlm> OLLAMA_HOST=... $0"
  exit 0
fi

MODEL="${VIDEO_SMOKE_MODEL:-}"
if [[ -z "${MODEL}" ]]; then
  echo "VIDEO_SMOKE_MODEL is required for RUN_E2E_VIDEO_AGENT=1" >&2
  exit 1
fi

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:8080}"
OLLAMA_URL="${OLLAMA_URL%/}"
CACHE_KEY="${VIDEO_AGENT_CACHE_KEY:-video-agent-smoke-1}"
GO_LOG="${VIDEO_AGENT_GO_LOG:-}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found — required for live video agent smoke" >&2
  exit 1
fi

if ! curl -sf -m 5 "${OLLAMA_URL}/api/tags" >/dev/null; then
  echo "zerollama not reachable at ${OLLAMA_URL} — start serve first" >&2
  exit 1
fi

echo "== live two-turn video agent cache (${MODEL}) =="
export OLLAMA_URL MODEL CACHE_KEY GO_LOG
python3 <<'PY'
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

url = os.environ["OLLAMA_URL"].rstrip("/")
model = os.environ["MODEL"]
cache_key = os.environ["CACHE_KEY"]
go_log = os.environ.get("GO_LOG", "").strip()

tmpdir = tempfile.mkdtemp(prefix="video-agent-smoke-")
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


def post_chat(messages):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "_debug_render_only": True,
        "options": {
            "prompt_cache_key": cache_key,
            "num_ctx": 8192,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if e.fp else ""
        raise SystemExit(f"chat HTTP {e.code}: {body[:800]}") from e


video_msg = {
    "role": "user",
    "content": "describe this clip",
    "videos": [video_b64],
}

out1 = post_chat([video_msg])
dbg1 = out1.get("_debug_info") or out1.get("debug_info")
img1 = (dbg1 or {}).get("image_count", 0)
if img1 < 1:
    raise SystemExit(f"turn1: expected expanded frames, got image_count={img1!r}")

out2 = post_chat(
    [
        video_msg,
        {"role": "assistant", "content": "test pattern"},
        {"role": "user", "content": "same clip again", "videos": [video_b64]},
    ]
)
dbg2 = out2.get("_debug_info") or out2.get("debug_info")
img2 = (dbg2 or {}).get("image_count", 0)
if img2 < 1:
    raise SystemExit(f"turn2: expected expanded frames, got image_count={img2!r}")

if go_log:
    try:
        with open(go_log, "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()
    except OSError as e:
        raise SystemExit(f"cannot read VIDEO_AGENT_GO_LOG={go_log}: {e}") from e
    if "video sample session cache hit" not in log_text:
        raise SystemExit(
            "turn2 completed but VIDEO_AGENT_GO_LOG lacks "
            "'video sample session cache hit' — set VIDEO_AGENT_GO_LOG to serve log path"
        )
    print("log: session cache hit confirmed")
    if "vision embed session cache hit" in log_text:
        print("log: vision embed session cache hit confirmed")
    else:
        print(
            "warn: VIDEO_AGENT_GO_LOG lacks 'vision embed session cache hit' "
            "(optional on turn2 if ViT overlay not exercised)",
            file=sys.stderr,
        )
else:
    print(
        "warn: VIDEO_AGENT_GO_LOG unset — turn2 OK but session cache log not verified",
        file=sys.stderr,
    )

print(json.dumps({"turn1_image_count": img1, "turn2_image_count": img2, "cache_key": cache_key}))

# OpenAI /v1/chat/completions path — same clip + prompt_cache_key (Tier 1 OpenAI wiring).
v1_payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "same clip v1"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + video_b64}},
            ],
        }
    ],
    "stream": False,
    "prompt_cache_key": cache_key,
    "_debug_render_only": True,
    "options": {"num_ctx": 8192},
}
v1_data = json.dumps(v1_payload).encode()
v1_req = urllib.request.Request(
    f"{url}/v1/chat/completions",
    data=v1_data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(v1_req, timeout=300) as resp:
        v1_out = json.loads(resp.read().decode())
except urllib.error.HTTPError as e:
    body = e.read().decode(errors="replace") if e.fp else ""
    raise SystemExit(f"v1 chat HTTP {e.code}: {body[:800]}") from e
v1_dbg = (v1_out.get("_debug_info") or v1_out.get("debug_info") or {})
v1_img = v1_dbg.get("image_count", 0)
if v1_img < 1:
    raise SystemExit(f"v1: expected expanded frames, got image_count={v1_img!r}")
print(json.dumps({"v1_image_count": v1_img, "cache_key": cache_key}))

# Pre-expanded SGLang path — images + video_spans + padded_input_ids turn 1; layout restore turn 2.
framedir = os.path.join(tmpdir, "frames")
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
    raise SystemExit(f"expected 2 PNG frames, got {len(frame_b64)}")

preproc_key = cache_key + "-preprocessed"
padded = [101, 102, 103, 104, 105]
preproc_span = [{"frame_count": len(frame_b64), "grid_thw": [len(frame_b64), 8, 8]}]


def post_preprocessed(messages, key):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "_debug_render_only": True,
        "options": {"prompt_cache_key": key, "num_ctx": 8192},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


pre1 = {
    "role": "user",
    "content": "preprocessed clip",
    "images": frame_b64,
    "video_spans": preproc_span,
    "padded_input_ids": padded,
}
out_pre1 = post_preprocessed([pre1], preproc_key)
dbg_pre1 = out_pre1.get("_debug_info") or out_pre1.get("debug_info") or {}
if dbg_pre1.get("padded_input_ids_len") != len(padded):
    raise SystemExit(
        f"preprocessed turn1: padded_input_ids_len={dbg_pre1.get('padded_input_ids_len')!r} "
        f"want {len(padded)}"
    )
if dbg_pre1.get("padded_layout_consume") not in (
    "deferred",
    "qwen3vl_hf_skip_placeholders",
    "qwen3vl_hf_runner_inject",
    "gemma3_img_skip_placeholders",
    "gemma3_img_runner_inject",
    "llama4_img_skip_placeholders",
    "llama4_img_runner_inject",
    "lfm2_img_skip_placeholders",
    "lfm2_img_runner_inject",
    "glmocr_img_skip_placeholders",
    "glmocr_img_runner_inject",
    "mistral3_img_skip_placeholders",
    "mistral3_img_runner_inject",
    "deepseekocr_img_skip_placeholders",
    "deepseekocr_img_runner_inject",
    "gemma4_img_skip_placeholders",
    "gemma4_img_runner_inject",
    "mllama_img_skip_placeholders",
    "mllama_img_runner_inject",
):
    raise SystemExit(
        f"preprocessed turn1: padded_layout_consume={dbg_pre1.get('padded_layout_consume')!r}"
    )

pre2_latest = {
    "role": "user",
    "content": "same preprocessed clip",
    "images": frame_b64,
    "video_spans": preproc_span,
}
out_pre2 = post_preprocessed(
    [
        {"role": "user", "content": "preprocessed clip", "images": frame_b64, "video_spans": preproc_span},
        {"role": "assistant", "content": "ok"},
        pre2_latest,
    ],
    preproc_key,
)
dbg_pre2 = out_pre2.get("_debug_info") or out_pre2.get("debug_info") or {}
if dbg_pre2.get("padded_input_ids_len") != len(padded):
    raise SystemExit(
        f"preprocessed turn2: padded_input_ids_len={dbg_pre2.get('padded_input_ids_len')!r} "
        f"want {len(padded)} (session layout restore)"
    )

if go_log:
    if "preprocessed layout session cache hit" not in log_text:
        raise SystemExit(
            "preprocessed turn2 OK but VIDEO_AGENT_GO_LOG lacks "
            "'preprocessed layout session cache hit'"
        )
    print("log: preprocessed layout session cache hit confirmed")

print(json.dumps({
    "preprocessed_turn2_padded_len": dbg_pre2.get("padded_input_ids_len"),
    "preprocessed_cache_key": preproc_key,
}))
PY

echo "PASS video_agent_cache_smoke (live E2E)"
