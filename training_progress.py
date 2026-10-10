"""T3 training progress metrics (loss / tokens/s / VRAM).

Emitted on the job progress channel (poll + SSE). Pure helpers so unit tests
do not need a live Trainer or CUDA.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional


def training_vram_snapshot(torch_mod: Any | None = None) -> Dict[str, Any]:
    """Return allocated / reserved / free VRAM for the current CUDA device.

    Empty dict when torch is missing or CUDA is unavailable.
    """
    torch = torch_mod
    if torch is None:
        try:
            import torch as _torch

            torch = _torch
        except ImportError:
            return {}
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {}
    try:
        idx = int(torch.cuda.current_device())
        free_b, total_b = torch.cuda.mem_get_info(idx)
        alloc_b = int(torch.cuda.memory_allocated(idx))
        reserved_b = int(torch.cuda.memory_reserved(idx))
    except Exception:
        return {}
    return {
        "device_index": idx,
        "allocated_bytes": alloc_b,
        "reserved_bytes": reserved_b,
        "free_bytes": int(free_b),
        "total_bytes": int(total_b),
        "allocated_gib": round(alloc_b / (1024**3), 3),
        "free_gib": round(int(free_b) / (1024**3), 3),
    }


def estimate_tokens_per_sec(
    *,
    steps_delta: int,
    elapsed_sec: float,
    samples_per_step: float,
    mean_tokens_per_sample: float,
) -> Optional[float]:
    """Rough train throughput: steps × samples/step × mean tokens / elapsed."""
    if steps_delta <= 0 or elapsed_sec <= 0:
        return None
    if samples_per_step <= 0 or mean_tokens_per_sample <= 0:
        return None
    return (steps_delta * samples_per_step * mean_tokens_per_sample) / elapsed_sec


def mean_tokens_per_sample(tokenized: Any, *, fallback: float) -> float:
    """Average sequence length from a HF Dataset / dict with ``input_ids``."""
    try:
        ids = tokenized["input_ids"]
    except Exception:
        return float(fallback)
    total = 0
    n = 0
    try:
        for row in ids:
            total += len(row)
            n += 1
    except Exception:
        return float(fallback)
    if n <= 0:
        return float(fallback)
    return float(total) / float(n)


def build_progress_metrics(
    *,
    step: int,
    total_steps: int,
    loss: Optional[float] = None,
    tokens_per_sec: Optional[float] = None,
    learning_rate: Optional[float] = None,
    vram: Optional[Mapping[str, Any]] = None,
    epoch: Optional[float] = None,
) -> Dict[str, Any]:
    """Structured metrics blob for job status / SSE ``data``."""
    out: Dict[str, Any] = {
        "step": int(step),
        "total_steps": int(total_steps),
    }
    if loss is not None:
        try:
            out["loss"] = float(loss)
        except (TypeError, ValueError):
            pass
    if tokens_per_sec is not None:
        try:
            out["tokens_per_sec"] = float(tokens_per_sec)
        except (TypeError, ValueError):
            pass
    if learning_rate is not None:
        try:
            out["learning_rate"] = float(learning_rate)
        except (TypeError, ValueError):
            pass
    if epoch is not None:
        try:
            out["epoch"] = float(epoch)
        except (TypeError, ValueError):
            pass
    if vram:
        out["vram"] = dict(vram)
    return out


def format_metrics_message(metrics: Mapping[str, Any]) -> str:
    """Human one-liner for ``progress_message`` (poll clients without metrics)."""
    step = metrics.get("step")
    total = metrics.get("total_steps")
    parts = []
    if step is not None and total:
        parts.append(f"step {step}/{total}")
    elif step is not None:
        parts.append(f"step {step}")
    loss = metrics.get("loss")
    if loss is not None:
        parts.append(f"loss={float(loss):.4f}")
    tps = metrics.get("tokens_per_sec")
    if tps is not None:
        parts.append(f"{float(tps):.0f} tok/s")
    vram = metrics.get("vram") or {}
    if isinstance(vram, Mapping) and vram.get("allocated_gib") is not None:
        parts.append(f"vram={vram['allocated_gib']:.2f}GiB")
    return " ".join(parts) if parts else "training"


class StepMetricsTracker:
    """Accumulate loss / tok/s between Trainer log events."""

    def __init__(
        self,
        *,
        total_steps: int,
        samples_per_step: float,
        mean_tokens_per_sample: float,
    ) -> None:
        self.total_steps = max(0, int(total_steps))
        self.samples_per_step = float(samples_per_step)
        self.mean_tokens_per_sample = float(mean_tokens_per_sample)
        self._last_t = time.monotonic()
        self._last_step = 0

    def on_step(
        self,
        *,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch: Optional[float] = None,
        torch_mod: Any | None = None,
    ) -> Dict[str, Any]:
        now = time.monotonic()
        steps_delta = max(0, int(step) - self._last_step)
        elapsed = now - self._last_t
        tps = estimate_tokens_per_sec(
            steps_delta=steps_delta,
            elapsed_sec=elapsed,
            samples_per_step=self.samples_per_step,
            mean_tokens_per_sample=self.mean_tokens_per_sample,
        )
        self._last_t = now
        self._last_step = int(step)
        return build_progress_metrics(
            step=step,
            total_steps=self.total_steps,
            loss=loss,
            tokens_per_sec=tps,
            learning_rate=learning_rate,
            epoch=epoch,
            vram=training_vram_snapshot(torch_mod),
        )
