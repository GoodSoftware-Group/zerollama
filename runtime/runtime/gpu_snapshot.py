"""Phase 13 snapshot → operator recommendations (5080 session JSON).

WHY this module exists: ``/health`` and ``gpu_health_report.sh`` are live-only. The snapshot
JSON from ``gpu_phase13_snapshot.sh`` is a portable tuning record; this turns it into env hints
without re-querying the server. Rules favor per-GGUF autotune persist over a global
``VRAM_ESTIMATE_FACTOR`` (smoke calibration on one GGUF must not be copied to all models).
Harmony real-weight is explicitly out of scope on ~19GiB host-RAM boxes — CI Go golden covers
parser behavior instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def recommend_from_snapshot(snap: dict[str, Any]) -> list[str]:
    """Actionable env hints from ``gpu_phase13_snapshot.sh`` / session JSON."""
    lines: list[str] = []
    vc = snap.get("vram_calibration") or {}
    va = snap.get("vram_autotune") or {}
    vb = snap.get("vram_budget") or {}
    veb = snap.get("vram_estimate_budget") or {}
    ac = snap.get("autoconfig") or {}
    ad = snap.get("admission") or {}
    gguf = snap.get("gguf")

    if ac.get("pick") == "single_gpu":
        lines.append(
            "# autoconfig: single_gpu.yaml (16GB-class); env overrides YAML vram: defaults"
        )

    vb_fits = vb.get("fits_with_margin")
    veb_fits = veb.get("fits_with_margin")
    if vb_fits is False or veb_fits is False:
        lines.append(
            "# WARN: fits_with_margin=false — lower num_ctx or use a smaller quant before serve"
        )

    autotune_on = bool(va.get("enabled"))
    eff = va.get("effective_factor")
    suggest = vc.get("suggested_estimate_factor")
    persist = va.get("persist") or {}
    persisted = persist.get("persisted_factor")
    catalog = persist.get("catalog") or []

    if autotune_on and catalog:
        lines.append(f"# autotune catalog: {len(catalog)} GGUF(s) calibrated")
        for row in catalog[:8]:
            if not isinstance(row, dict):
                continue
            name = row.get("basename") or row.get("model")
            factor = row.get("estimate_factor")
            if name is None or factor is None:
                continue
            suffix = " (last)" if row.get("last") else ""
            lines.append(f"#   {name}: factor {float(factor):g}{suffix}")
        if gguf:
            try:
                from runtime.vram_autotune_persist import model_in_persist_catalog

                if not model_in_persist_catalog(gguf):
                    lines.append(
                        f"# probe GGUF not in catalog — run one probed load for {Path(gguf).name!r}"
                    )
            except (OSError, ValueError):
                pass
        if persisted is not None:
            lines.append(
                "# per-GGUF persist wins; no global ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR needed"
            )
    elif autotune_on and eff is not None:
        model = va.get("session_model") or vc.get("model") or gguf
        if model:
            lines.append(f"# autotune active for {model!r} (factor {eff:g})")
        else:
            lines.append(f"# autotune effective_factor={eff:g}")
        if persisted is not None:
            lines.append(
                "# per-GGUF persist wins; no global ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR needed"
            )
        else:
            lines.append(
                "# run one load per production GGUF to seed vram_autotune.json"
            )
    elif suggest is not None:
        try:
            sf = float(suggest)
            if 0.1 <= sf <= 3.0:
                lines.append(f"export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR={sf:g}")
                lines.append(
                    "# or: ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE=auto after probe"
                )
            else:
                lines.append(
                    f"# suggested_estimate_factor={sf:g} outside 0.1–3 — review calibration"
                )
        except (TypeError, ValueError):
            pass

    smax = vb.get("suggested_max_num_ctx") or veb.get("suggested_max_num_ctx")
    if smax:
        lines.append(f"# suggested_max_num_ctx={smax} (probe at num_ctx={snap.get('num_ctx_probe')})")
        lines.append("# optional tight serve: ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto")

    if ad.get("vram_min_free_configured") is not None:
        lines.append(
            "# Phase 11 headroom configured at snapshot "
            f"(min_free={ad.get('vram_min_free_configured')}, "
            f"training_reserve={ad.get('vram_training_reserve_configured')})"
        )

    # Harmony / gpt-oss:20b not validated on ~19GiB host RAM — CI Go golden covers parser.
    lines.append(
        "# harmony real-weight capture: skip on ~19GiB host; use ./scripts/phase12_golden_ci.sh"
    )

    return lines


def format_snapshot_recommendations(snap: dict[str, Any]) -> str:
    lines = recommend_from_snapshot(snap)
    if not lines:
        return "== Phase 13 snapshot recommendations ==\n(no recommendations)"
    body = "\n".join(lines)
    return f"== Phase 13 snapshot recommendations ==\n{body}"


def load_snapshot(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("snapshot root must be an object")
    return data


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Print VRAM tuning hints from gpu_phase13_snapshot JSON"
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="snapshot JSON (default: GPU_PHASE13_SNAPSHOT_OUT or stdin)",
    )
    args = parser.parse_args()
    if args.file is not None:
        snap = load_snapshot(args.file)
    else:
        import os

        env_path = os.environ.get("GPU_PHASE13_SNAPSHOT_OUT", "").strip()
        if env_path:
            snap = load_snapshot(Path(env_path))
        elif not sys.stdin.isatty():
            snap = json.loads(sys.stdin.read())
        else:
            parser.error("provide a snapshot file, GPU_PHASE13_SNAPSHOT_OUT, or stdin")
    print(format_snapshot_recommendations(snap))


if __name__ == "__main__":
    main()
