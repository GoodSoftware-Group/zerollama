"""Map runtime config to llama.cpp speculative CLI flags (one active method)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Aliases from zerollama config names → llama.cpp --spec-type values.
METHOD_ALIASES: dict[str, str] = {
    "none": "none",
    "ngram": "ngram-simple",
    "ngram-simple": "ngram-simple",
    "ngram-mod": "ngram-mod",
    "ngram-cache": "ngram-cache",
    "draft": "draft-simple",
    "draft-simple": "draft-simple",
    "eagle3": "draft-eagle3",
    "draft-eagle3": "draft-eagle3",
    "dflash": "dflash",
    "mtp": "draft-mtp",
    "draft-mtp": "draft-mtp",
}


@dataclass
class SpeculativeConfig:
    method: str = "none"
    draft_model: Path | None = None
    draft_n_max: int = 16
    draft_n_min: int = 0
    draft_n_gpu_layers: int = -1
    ngram_size_n: int = 12
    ngram_size_m: int = 48
    ngram_min_hits: int = 1

    @classmethod
    def from_mapping(cls, spec: dict[str, Any] | None) -> SpeculativeConfig:
        if not spec:
            return cls()
        ngram = spec.get("ngram") or {}
        draft = spec.get("draft") or {}
        draft_path = spec.get("draft_model")
        return cls(
            method=str(spec.get("method", "none")),
            draft_model=Path(draft_path) if draft_path else None,
            draft_n_max=int(draft.get("n_max", spec.get("draft_n_max", 16))),
            draft_n_min=int(draft.get("n_min", spec.get("draft_n_min", 0))),
            draft_n_gpu_layers=int(draft.get("n_gpu_layers", -1)),
            ngram_size_n=int(ngram.get("size_n", spec.get("ngram_size_n", 12))),
            ngram_size_m=int(ngram.get("size_m", spec.get("ngram_size_m", 48))),
            ngram_min_hits=int(ngram.get("min_hits", spec.get("ngram_min_hits", 1))),
        )


def resolve_method(name: str) -> str:
    key = name.strip().lower()
    return METHOD_ALIASES.get(key, key)


def llama_server_args_for(spec: SpeculativeConfig) -> list[str]:
    """Return extra llama-server argv for the active speculative plugin."""
    llama_type = resolve_method(spec.method)
    if llama_type == "none":
        return []

    args: list[str] = ["--spec-type", llama_type]

    if llama_type.startswith("ngram"):
        args.extend(
            [
                "--spec-ngram-simple-size-n",
                str(spec.ngram_size_n),
                "--spec-ngram-simple-size-m",
                str(spec.ngram_size_m),
                "--spec-ngram-simple-min-hits",
                str(spec.ngram_min_hits),
            ]
        )
        return args

    if llama_type.startswith("draft"):
        draft = spec.draft_model
        if draft is None or not draft.is_file():
            raise ValueError(
                f"speculative method {spec.method!r} requires draft_model path"
            )
        args.extend(["--model-draft", str(draft)])
        if spec.draft_n_max > 0:
            args.extend(["--spec-draft-n-max", str(spec.draft_n_max)])
        if spec.draft_n_min > 0:
            args.extend(["--spec-draft-n-min", str(spec.draft_n_min)])
        args.extend(["--spec-draft-ngl", str(spec.draft_n_gpu_layers)])
        return args

    args.extend(["--spec-type", llama_type])
    return args
