"""Ollama-shaped sampling options for llama-server HTTP and in-process libllama.

Why a shared module: Phase 14 has three forward paths (subprocess JSON, ctypes sampler
chain, llama-cpp-python kwargs). Divergent defaults would make smokes flaky and break
parity with api.DefaultOptions on the Go side.

Why greedy when no keys: empty options on in-process must stay deterministic for GPU
smokes; subprocess keeps llama-server's own defaults when we omit fields.

Why full Ollama defaults when any key is set: partial client options (e.g. only
temperature) should not leave top_k/top_p at library-internal zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Keys forwarded to llama-server / used to build in-process sampler chains.
_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "seed",
        "repeat_penalty",
        "repeat_last_n",
        "presence_penalty",
        "frequency_penalty",
    }
)

# Ollama api.DefaultOptions inference defaults (api/types.go).
_OLLAMA_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": -1,
}


@dataclass(frozen=True)
class SamplerOptions:
    """Resolved sampling parameters for one completion."""

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = -1
    greedy_only: bool = False

    @property
    def dist_seed(self) -> int:
        """Seed for llama_sampler_init_dist (``-1`` → random)."""
        if self.seed < 0:
            return 0xFFFFFFFF
        return int(self.seed) & 0xFFFFFFFF


def sampler_options_from_dict(options: dict[str, Any] | None) -> SamplerOptions | None:
    """Return ``None`` when the request has no sampling keys (legacy greedy in-process)."""
    if not options:
        return None
    present = _SAMPLING_KEYS.intersection(options.keys())
    if not present:
        return None
    vals = dict(_OLLAMA_DEFAULTS)
    for key in present:
        vals[key] = _coerce_option(key, options[key])
    temp = float(vals["temperature"])
    if temp <= 0.0:
        return SamplerOptions(greedy_only=True)
    return SamplerOptions(
        temperature=temp,
        top_k=int(vals["top_k"]),
        top_p=float(vals["top_p"]),
        min_p=float(vals["min_p"]),
        typical_p=float(vals["typical_p"]),
        repeat_penalty=float(vals["repeat_penalty"]),
        repeat_last_n=int(vals["repeat_last_n"]),
        presence_penalty=float(vals["presence_penalty"]),
        frequency_penalty=float(vals["frequency_penalty"]),
        seed=int(vals["seed"]),
        greedy_only=False,
    )


def apply_sampler_to_completion_payload(
    payload: dict[str, Any],
    sampler: SamplerOptions | None,
) -> None:
    """Merge sampling fields into llama-server ``/completion`` JSON."""
    if sampler is None:
        return
    if sampler.greedy_only:
        # Match in-process greedy chain: llama temp sampler with t<=0 keeps argmax logit.
        payload["temperature"] = 0.0
        return
    payload["temperature"] = sampler.temperature
    payload["top_k"] = sampler.top_k
    payload["top_p"] = sampler.top_p
    if sampler.min_p > 0:
        payload["min_p"] = sampler.min_p
    if sampler.typical_p < 1.0:
        payload["typical_p"] = sampler.typical_p
    payload["repeat_penalty"] = sampler.repeat_penalty
    payload["repeat_last_n"] = sampler.repeat_last_n
    if sampler.presence_penalty != 0.0:
        payload["presence_penalty"] = sampler.presence_penalty
    if sampler.frequency_penalty != 0.0:
        payload["frequency_penalty"] = sampler.frequency_penalty
    if sampler.seed >= 0:
        payload["seed"] = sampler.seed


def sampler_to_llama_cpp_kwargs(sampler: SamplerOptions | None) -> dict[str, Any]:
    """Kwargs for ``llama_cpp.Llama.create_completion``."""
    if sampler is None or sampler.greedy_only:
        return {"temperature": 0.0}
    out: dict[str, Any] = {
        "temperature": sampler.temperature,
        "top_k": sampler.top_k,
        "top_p": sampler.top_p,
        "min_p": sampler.min_p,
        "typical_p": sampler.typical_p,
        "repeat_penalty": sampler.repeat_penalty,
        "frequency_penalty": sampler.frequency_penalty,
        "presence_penalty": sampler.presence_penalty,
    }
    if sampler.seed >= 0:
        out["seed"] = sampler.seed
    return out


def _coerce_option(key: str, raw: Any) -> float | int:
    if raw is None:
        return _OLLAMA_DEFAULTS[key]  # type: ignore[return-value]
    if key in ("top_k", "repeat_last_n", "seed"):
        return int(raw)
    return float(raw)
