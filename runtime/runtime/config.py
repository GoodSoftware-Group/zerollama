"""Runtime configuration from YAML file and environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runtime.speculative import SpeculativeConfig, llama_server_args_for


def _zerollama_root() -> Path:
    # runtime/runtime/config.py -> zerollama repo root
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "dual_4090.yaml"


_DEFAULT_LLAMA_BACKEND = "subprocess"


def _normalize_llama_backend(raw: str | None) -> str:
    """Canonical backend string for RuntimeConfig."""
    from runtime.worker.factory import canonical_llama_backend

    return canonical_llama_backend(raw, default=_DEFAULT_LLAMA_BACKEND)


@dataclass
class RuntimeConfig:
    host: str
    port: int
    llama_cpp_root: Path
    llama_server_bin: Path | None
    llama_model: Path | None
    num_blocks: int
    block_size: int
    device_count: int
    tensor_parallel: int = 1
    split_mode: str = "tensor"
    tensor_split: tuple[float, ...] | None = None
    main_gpu: int = 0
    speculative_method: str = "none"
    draft_model: Path | None = None
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    llama_parallel_slots: int = 4
    llama_backend: str = _DEFAULT_LLAMA_BACKEND
    llama_backend_from_file: bool = False
    llama_cpp_lib: Path | None = None

    def active_kv_pools(self) -> int:
        """How many device pools participate in KV for one sequence."""
        if self.tensor_parallel > 1:
            return min(self.tensor_parallel, self.device_count)
        return 1

    def llama_server_args(self) -> list[str]:
        """Extra flags for llama-server subprocess."""
        split_mode = self.split_mode
        # Why force layer on single GPU: llama-server tensor split (-sm tensor -ts …) requires
        # multiple devices; on one card fitting fails with SPLIT_MODE_TENSOR (see smoke doc).
        if self.tensor_parallel <= 1:
            split_mode = "layer"
        args: list[str] = [
            "-sm",
            split_mode,
            "-mg",
            str(self.main_gpu),
            "-np",
            str(self.llama_parallel_slots),
        ]
        if self.tensor_parallel > 1 and split_mode in ("tensor", "row", "layer"):
            splits = self.tensor_split
            if splits is None:
                splits = tuple(1.0 for _ in range(self.tensor_parallel))
            args.extend(["-ts", ",".join(str(s) for s in splits)])
        try:
            args.extend(llama_server_args_for(self.speculative))
        except ValueError:
            if self.speculative.method != "none":
                raise
        extra = os.environ.get("LLAMA_SERVER_EXTRA_ARGS", "").strip()
        if extra:
            args.extend(extra.split())
        return args

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        cfg_path = os.environ.get("ZEROLLAMA_RUNTIME_CONFIG", "").strip()
        if cfg_path:
            return cls.from_file(Path(cfg_path))
        from runtime.autoconfig import resolve_default_config_path

        return cls.from_file(resolve_default_config_path())

    @classmethod
    def from_file(cls, path: Path) -> RuntimeConfig:
        data = _load_yaml(path)
        return cls._from_mapping(data)._apply_env_overrides()

    @classmethod
    def _defaults_from_env_only(cls) -> RuntimeConfig:
        root = Path(
            os.environ.get("LLAMA_CPP_ROOT", _zerollama_root().parent / "llama.cpp")
        )
        return cls(
            host=os.environ.get("ZEROLLAMA_RUNTIME_HOST", "127.0.0.1"),
            port=int(os.environ.get("ZEROLLAMA_RUNTIME_PORT", "8081")),
            llama_cpp_root=root,
            llama_server_bin=_resolve_llama_bin(root),
            llama_model=_resolve_model(),
            num_blocks=int(os.environ.get("ZEROLLAMA_KV_NUM_BLOCKS", "4096")),
            block_size=int(os.environ.get("ZEROLLAMA_KV_BLOCK_SIZE", "16")),
            device_count=int(os.environ.get("ZEROLLAMA_DEVICE_COUNT", "1")),
            tensor_parallel=int(os.environ.get("ZEROLLAMA_TENSOR_PARALLEL", "1")),
        )._apply_env_overrides()

    @classmethod
    def _from_mapping(cls, data: dict[str, Any]) -> RuntimeConfig:
        kv = data.get("kv") or {}
        spec = data.get("speculative") or {}
        root = Path(
            os.environ.get("LLAMA_CPP_ROOT", _zerollama_root().parent / "llama.cpp")
        )
        device_count = int(data.get("device_count", 2))
        tp = int(data.get("tensor_parallel", 1))
        split_mode = str(data.get("split_mode", "tensor" if tp > 1 else "layer"))
        ts_raw = data.get("tensor_split")
        tensor_split: tuple[float, ...] | None = None
        if ts_raw is not None:
            tensor_split = tuple(float(x) for x in ts_raw)

        draft_s = spec.get("draft_model") or data.get("draft_model")
        draft = Path(draft_s) if draft_s else None
        spec_cfg = SpeculativeConfig.from_mapping(spec)
        if draft is not None:
            spec_cfg.draft_model = draft

        backend_raw = data.get("llama_backend")
        backend_from_file = backend_raw is not None and str(backend_raw).strip() != ""

        return cls(
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 8081)),
            llama_cpp_root=root,
            llama_server_bin=_resolve_llama_bin(root),
            llama_model=_resolve_model(data.get("llama_model")),
            num_blocks=int(
                kv.get(
                    "num_blocks_per_device",
                    kv.get("num_blocks", 4096),
                )
            ),
            block_size=int(kv.get("block_size", 16)),
            device_count=device_count,
            tensor_parallel=tp,
            split_mode=split_mode,
            tensor_split=tensor_split,
            main_gpu=int(data.get("main_gpu", 0)),
            speculative_method=spec_cfg.method,
            draft_model=spec_cfg.draft_model or draft,
            speculative=spec_cfg,
            llama_parallel_slots=int(data.get("llama_parallel_slots", 4)),
            llama_backend=_normalize_llama_backend(backend_raw),
            llama_backend_from_file=backend_from_file,
        )

    def _apply_env_overrides(self) -> RuntimeConfig:
        """Env wins over file defaults (operator overrides)."""
        if v := os.environ.get("ZEROLLAMA_RUNTIME_HOST"):
            self.host = v
        if v := os.environ.get("ZEROLLAMA_RUNTIME_PORT"):
            self.port = int(v)
        if v := os.environ.get("ZEROLLAMA_KV_NUM_BLOCKS"):
            self.num_blocks = int(v)
        if v := os.environ.get("ZEROLLAMA_KV_BLOCK_SIZE"):
            self.block_size = int(v)
        if v := os.environ.get("ZEROLLAMA_DEVICE_COUNT"):
            self.device_count = int(v)
        if v := os.environ.get("ZEROLLAMA_TENSOR_PARALLEL"):
            self.tensor_parallel = int(v)
        if v := os.environ.get("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "").strip():
            self.llama_backend = _normalize_llama_backend(v)
        lib_env = os.environ.get("LLAMA_CPP_LIB", "").strip()
        if lib_env:
            self.llama_cpp_lib = Path(lib_env)
        root = self.llama_cpp_root
        bin_p = _resolve_llama_bin(root)
        if bin_p is not None:
            self.llama_server_bin = bin_p
        model = _resolve_model()
        if model is not None:
            self.llama_model = model
        if v := os.environ.get("ZEROLLAMA_SPEC_METHOD"):
            self.speculative_method = v
            self.speculative.method = v
        draft_env = os.environ.get("LLAMA_DRAFT_MODEL", "").strip()
        if draft_env:
            p = Path(draft_env)
            if p.is_file():
                self.draft_model = p
                self.speculative.draft_model = p
        return self


def _resolve_llama_bin(root: Path) -> Path | None:
    override = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    candidate = Path(override) if override else root / "build" / "bin" / "llama-server"
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    return candidate if candidate.is_file() else None


def _resolve_model(yaml_path: Any = None) -> Path | None:
    if yaml_path:
        p = Path(yaml_path)
        if not p.is_absolute():
            p = p.resolve()
        return p if p.is_file() else None
    model_s = os.environ.get("LLAMA_MODEL", "").strip()
    if not model_s:
        return None
    p = Path(model_s)
    if not p.is_absolute():
        p = p.resolve()
    return p if p.is_file() else None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML required for config files: pip install pyyaml"
        ) from e
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw
