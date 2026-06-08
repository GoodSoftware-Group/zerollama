"""Format /health JSON for GPU operator tuning (used by scripts/gpu_health_report.sh)."""

from __future__ import annotations

from typing import Any

from runtime.vram_recommendations import skip_global_vram_factor_export


def format_gpu_health_tuning_report(h: dict[str, Any]) -> str:
    """Return human-readable Phase 11–13 tuning lines for a /health payload."""
    lines: list[str] = []
    lines.append("== runtime /health (GPU tuning) ==")
    lines.append(f"status: {h.get('status')}")
    lines.append(f"llama_server: {h.get('llama_server')}")
    lines.append(f"inference_state: {h.get('inference_state')}")
    if h.get("llama_backend"):
        lines.append(f"llama_backend: {h.get('llama_backend')}")
    ac = h.get("autoconfig") or {}
    if h.get("llama_backend_source"):
        lines.append(f"llama_backend_source: {h.get('llama_backend_source')}")
        if h.get("llama_backend_source") == "default" and h.get("llama_backend") == "subprocess":
            cfg_path = ac.get("config_path")
            if cfg_path:
                lines.append(
                    f"# subprocess default via {cfg_path} (no llama_backend key in YAML)"
                )
            else:
                lines.append(
                    "# subprocess packaged default (no llama_backend key in autoconfig YAML)"
                )
    if h.get("llama_backend_requested"):
        lines.append(f"llama_backend_requested: {h.get('llama_backend_requested')}")
    if h.get("llama_backend_fallback"):
        lines.append("llama_backend_fallback: true")
        lines.append(
            "# inprocess load failed; subprocess llama-server active "
            "(ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK)"
        )

    if ac:
        lines.append(f"autoconfig: {ac.get('pick')} {ac.get('config_path')}")
        if (
            h.get("llama_backend_source") == "config"
            and h.get("llama_backend")
            and ac.get("config_path")
        ):
            lines.append(
                f"# llama_backend={h.get('llama_backend')} from {ac.get('config_path')}"
            )

    lcp = h.get("llama_cpp") or {}
    if lcp and h.get("llama_backend") == "llama-cpp-python":
        lines.append(f"llama_cpp.gpu_mode: {lcp.get('gpu_mode')}")
        lines.append(f"llama_cpp.n_gpu_layers: {lcp.get('n_gpu_layers')}")
        if lcp.get("env_n_gpu_layers"):
            lines.append(f"llama_cpp.env_n_gpu_layers: {lcp.get('env_n_gpu_layers')}")
        if lcp.get("gpu_mode") == "cpu" and not lcp.get("env_n_gpu_layers"):
            lines.append(
                "# wheel on CPU by default; set ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS after "
                "ctypes inprocess smoke passes on this host"
            )

    vb = h.get("vram_budget") or {}
    for k in (
        "admission_fits",
        "fits_with_margin",
        "suggested_max_num_ctx",
        "num_ctx_over_budget",
    ):
        if k in vb:
            lines.append(f"vram_budget.{k}: {vb.get(k)}")

    ve = h.get("vram_estimate") or {}
    if ve.get("estimate_factor_source"):
        lines.append(
            f"vram_estimate.estimate_factor_source: {ve.get('estimate_factor_source')}"
        )
    if ve.get("estimate_factor_effective") is not None:
        lines.append(
            f"vram_estimate.estimate_factor_effective: {ve.get('estimate_factor_effective')}"
        )

    cal = h.get("vram_calibration") or {}
    if cal:
        if cal.get("model"):
            lines.append(f"vram_calibration.model: {cal.get('model')}")
        for k in (
            "suggested_estimate_factor",
            "observed_bytes",
            "estimated_raw_bytes",
            "active_estimate_factor",
            "age_s",
        ):
            if cal.get(k) is not None:
                lines.append(f"vram_calibration.{k}: {cal.get(k)}")

    at = h.get("vram_autotune") or {}
    autotune_persist = at.get("persist") or {}
    autotune_catalog = autotune_persist.get("catalog") or []
    if at:
        lines.append(f"vram_autotune.enabled: {at.get('enabled')}")
        lines.append(
            f"vram_autotune.pending_first_calibration: {at.get('pending_first_calibration')}"
        )
        for k in ("session_model", "session_factor", "effective_factor", "env_factor"):
            if at.get(k) is not None:
                lines.append(f"vram_autotune.{k}: {at.get(k)}")
        if at.get("probe_calibrate_required"):
            lines.append(
                f"vram_autotune.probe_calibrate_required: {at.get('probe_calibrate_required')}"
            )
        if autotune_persist.get("last_model"):
            lines.append(
                f"vram_autotune.persist.last_model: {autotune_persist.get('last_model')}"
            )
        if autotune_persist.get("persisted_factor") is not None:
            lines.append(
                "vram_autotune.persist.persisted_factor: "
                f"{autotune_persist.get('persisted_factor')}"
            )
        if autotune_persist.get("catalog_truncated"):
            lines.append("vram_autotune.persist.catalog_truncated: true")
        if autotune_catalog:
            lines.append(f"vram_autotune.persist.catalog_count: {len(autotune_catalog)}")
            for row in autotune_catalog[:5]:
                if not isinstance(row, dict):
                    continue
                name = row.get("basename") or row.get("model")
                factor = row.get("estimate_factor")
                last = row.get("last")
                suffix = " (last)" if last else ""
                lines.append(
                    f"vram_autotune.persist.catalog: {name} factor={factor}{suffix}"
                )

    policy = h.get("vram_num_ctx_policy") or {}
    if policy:
        lines.append(
            f"vram_num_ctx_policy: {policy.get('env')} "
            f"clamp_enabled= {policy.get('clamp_enabled')}"
        )

    ad = h.get("admission") or {}
    if ad.get("vram_min_free_configured") is not None:
        lines.append(
            f"admission.vram_min_free_configured: {ad.get('vram_min_free_configured')}"
        )
    if ad.get("vram_training_reserve_configured") is not None:
        lines.append(
            "admission.vram_training_reserve_configured: "
            f"{ad.get('vram_training_reserve_configured')}"
        )

    lines.append("")
    lines.append("Suggested next steps:")
    suggest = cal.get("suggested_estimate_factor")
    factor_source = ve.get("estimate_factor_source")
    skip_global_factor_export = skip_global_vram_factor_export(
        autotune_enabled=bool(at.get("enabled")),
        catalog=autotune_catalog,
        factor_source=factor_source,
        persisted_factor=autotune_persist.get("persisted_factor"),
    )
    if suggest is not None and not skip_global_factor_export:
        try:
            sf = float(suggest)
            if 0.1 <= sf <= 3.0:
                lines.append(f"  export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR={sf:g}")
            else:
                lines.append(
                    f"  # suggested_estimate_factor={sf:g} out of clamp range 0.1–3; review calibration"
                )
        except (TypeError, ValueError):
            lines.append(f"  # suggested_estimate_factor={suggest!r} (not a number)")
        lines.append("  # or rely on VRAM_ESTIMATE_FACTOR_AUTOTUNE=auto persist")
    elif suggest is not None and skip_global_factor_export:
        lines.append(
            "  # per-GGUF autotune active; no global VRAM_ESTIMATE_FACTOR export needed"
        )
    eff = at.get("effective_factor")
    if eff is not None and at.get("enabled"):
        lines.append(f"  # effective autotune factor now: {eff}")
    if vb.get("suggested_max_num_ctx") and not policy.get("clamp_enabled"):
        lines.append("  # optional: ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto")
        lines.append(f"  # request num_ctx <= {vb.get('suggested_max_num_ctx')}")
    if at.get("pending_first_calibration"):
        lines.append("  # run one GPU generate (RUN_E2E_GPU=1) to seed autotune")
    if at.get("probe_calibrate_required"):
        lines.append("  # enable ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE=auto for autotune")

    return "\n".join(lines)


def main() -> None:
    """CLI: read HEALTH_JSON env or stdin; print tuning report."""
    import json
    import os
    import sys

    raw = os.environ.get("HEALTH_JSON", "").strip()
    if not raw and not sys.stdin.isatty():
        raw = sys.stdin.read()
    if not raw:
        print("usage: HEALTH_JSON='{...}' python -m runtime.gpu_health_report", file=sys.stderr)
        raise SystemExit(2)
    h = json.loads(raw)
    print(format_gpu_health_tuning_report(h))


if __name__ == "__main__":
    main()
