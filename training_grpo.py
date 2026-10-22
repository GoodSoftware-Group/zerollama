"""T10 — GRPO / RL job kind on the stock training path (TRL).

WHY one backend: Unsloth's GRPO/Studio path is a second stack (and Studio is AGPL).
Zerollama keeps PEFT + Transformers/``trl`` on the same embed JOB_QUEUE as SFT so T6
idle-wait / defer and the VRAM broker stay one policy.

WHY dry-run by default: CI and hosts without ``trl`` still validate payloads; operators
must opt into ``grpo_run=true`` on a free GPU lab.

WHY HTTP only allows ``reward_builtin``: JSON cannot carry Python callables. Pretending
``reward_funcs`` works over the wire caused silent failures — reject with a clear error.
In-process callers of ``process_grpo_request`` may still pass a real callable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

RewardFunc = Callable[..., List[float]]


def normalize_grpo_prompts(data: Dict[str, Any]) -> List[str]:
    """Accept prompts[] or training_data[{prompt}|str]."""
    raw = data.get("prompts")
    if raw is None:
        raw = data.get("training_data")
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            p = item.get("prompt") or item.get("question") or item.get("input")
            if isinstance(p, str) and p.strip():
                out.append(p.strip())
    return out


def _truthy_flag(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return False


_KNOWN_BUILTINS = frozenset({"length", "len"})


def _builtin_name(data: Dict[str, Any]) -> Optional[str]:
    """Resolve HTTP-safe builtin name from reward_builtin / aliases."""
    for key in ("reward_builtin", "reward_model", "reward"):
        v = data.get(key)
        if v is True and key == "reward_builtin":
            return "length"
        if isinstance(v, str):
            name = v.strip().lower()
            if not name:
                continue
            if name in _KNOWN_BUILTINS:
                return "length"
    return None


def _unknown_builtin_error(data: Dict[str, Any]) -> Optional[str]:
    v = data.get("reward_builtin")
    if isinstance(v, str) and v.strip() and v.strip().lower() not in _KNOWN_BUILTINS:
        return f"unknown reward_builtin {v!r} (want length)"
    return None


def length_reward(completions: List[str], **kwargs: Any) -> List[float]:
    """Smoke-friendly builtin: reward = completion character length."""
    return [float(len(c) if c is not None else 0) for c in completions]


def resolve_reward_func(data: Dict[str, Any]) -> Tuple[Optional[RewardFunc], Optional[str]]:
    """Return (func, error). HTTP path: builtins only. In-process: allow callables."""
    for key in ("reward_funcs", "reward_fn", "reward"):
        v = data.get(key)
        if callable(v):
            return v, None  # type: ignore[return-value]

    name = _builtin_name(data)
    if name == "length":
        return length_reward, None

    unknown = _unknown_builtin_error(data)
    if unknown:
        return None, unknown

    # Explicit reject of JSON-only "callable" pretenders.
    for key in ("reward_funcs", "reward_fn"):
        if data.get(key) is not None and not callable(data.get(key)):
            return None, (
                f"{key} cannot be sent over JSON; use reward_builtin=length "
                "(or call process_grpo_request in-process with a Python callable)"
            )
    if isinstance(data.get("reward"), str) and data["reward"].strip().lower() not in (
        "length",
        "len",
        "",
    ):
        return None, (
            "reward string must be a builtin name (length); "
            "HF reward models are T10b"
        )
    rm = data.get("reward_model")
    if isinstance(rm, str) and rm.strip() and rm.strip().lower() not in _KNOWN_BUILTINS:
        return None, (
            "reward_model must be 'length' for now "
            "(HF reward checkpoints are T10b)"
        )

    return None, (
        "reward signal required: reward_builtin=length "
        "(HTTP) or a Python callable via reward_funcs (in-process)"
    )


def has_reward_signal(data: Dict[str, Any]) -> bool:
    func, err = resolve_reward_func(data)
    return func is not None and err is None


def validate_grpo_payload(data: Dict[str, Any]) -> List[str]:
    """Return human-readable validation errors (empty = ok)."""
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["payload must be a JSON object"]
    model = data.get("model_name") or data.get("model")
    if not model or not str(model).strip():
        errors.append("model_name required")
    prompts = normalize_grpo_prompts(data)
    if len(prompts) < 1:
        errors.append("prompts (or training_data with prompt fields) required")
    _, reward_err = resolve_reward_func(data)
    if reward_err:
        errors.append(reward_err)
    return errors


def trl_grpo_available() -> Tuple[bool, Optional[str]]:
    """Return (ok, error_hint)."""
    try:
        import trl  # noqa: F401
    except ImportError:
        return False, (
            "trl not installed in embed/.venv-training; "
            "pip install 'trl>=0.14' (same interpreter as training.py)"
        )
    try:
        from trl import GRPOTrainer, GRPOConfig  # noqa: F401

        return True, None
    except ImportError:
        try:
            from trl.trainer import GRPOTrainer  # noqa: F401

            return True, None
        except ImportError:
            return False, (
                "trl installed but GRPOTrainer missing; "
                "upgrade trl (pip install -U 'trl>=0.14')"
            )


def process_grpo_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate; dry-run by default; GRPOTrainer only when grpo_run=true.

    WHY not use a bare ``run`` flag: collides with unrelated payload fields; require
    ``grpo_run`` so automation cannot accidentally start a Trainer.
    """
    errors = validate_grpo_payload(request)
    if errors:
        return {"status": "error", "error": "; ".join(errors), "kind": "grpo"}

    prompts = normalize_grpo_prompts(request)
    model_name = request.get("model_name") or request.get("model")
    want_run = _truthy_flag(request.get("grpo_run"))

    trl_ok, trl_hint = trl_grpo_available()

    # Dry-run: payload OK; trl is advisory only (scaffold must work without trl).
    # WHY status=ok + trained=false: job completed as validation, not as a train.
    if not want_run:
        return {
            "status": "ok",
            "kind": "grpo",
            "dry_run": True,
            "trained": False,
            "trl": trl_ok,
            "trl_hint": None if trl_ok else trl_hint,
            "n_prompts": len(prompts),
            "model_name": model_name,
            "reward_builtin": _builtin_name(request) or "callable",
            "message": (
                "T10 dry-run: payload OK; no training performed. "
                "Set grpo_run=true on a free GPU lab (requires trl) to run GRPOTrainer."
            ),
        }

    if not trl_ok:
        return {
            "status": "error",
            "error": trl_hint or "trl unavailable",
            "kind": "grpo",
            "dry_run": False,
            "trained": False,
        }

    return _run_grpo_trainer(request)


def _as_train_dataset(prompts: List[str]) -> Any:
    rows = [{"prompt": p} for p in prompts]
    try:
        from datasets import Dataset

        return Dataset.from_list(rows)
    except Exception:
        return rows


def _run_grpo_trainer(request: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal GRPOTrainer wire — lab only (grpo_run=true).

    WHY caller owns ``training_active``: clearing it here opened a window where
    OccupiesGPU / block-inference saw idle before the worker finished the job.
    """
    import training as training_mod

    training_mod._ensure_training_stack()

    reward_func, reward_err = resolve_reward_func(request)
    if reward_func is None:
        return {
            "status": "error",
            "error": reward_err or "reward unresolved",
            "kind": "grpo",
            "trained": False,
        }

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        return {
            "status": "error",
            "error": f"trl import failed: {e}",
            "kind": "grpo",
            "trained": False,
        }

    model_name = str(request.get("model_name") or request.get("model"))
    output_dir = str(request.get("output_dir") or "/tmp/grpo_output")
    prompts = normalize_grpo_prompts(request)

    training_mod.STATE.send_progress(5.0, "GRPO: loading model...")
    if not training_mod.STATE.load_model(
        model_name,
        use_lora=bool(request.get("use_lora", True)),
        use_qlora=bool(request.get("use_qlora", False)),
        lora_rank=int(request.get("lora_rank", 16) or 16),
        lora_alpha=float(request.get("lora_alpha", 32.0) or 32.0),
        lora_request=request,
    ):
        return {
            "status": "error",
            "error": "Failed to load model",
            "kind": "grpo",
            "trained": False,
        }

    max_steps = int(request.get("max_steps", 10) or 10)
    try:
        args = GRPOConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=int(request.get("batch_size", 1) or 1),
            logging_steps=1,
            report_to=[],
        )
    except TypeError as e:
        return {
            "status": "error",
            "error": f"GRPOConfig API mismatch: {e}",
            "kind": "grpo",
            "trained": False,
        }

    dataset = _as_train_dataset(prompts)
    training_mod.STATE.send_progress(20.0, "GRPO: training...")

    try:
        trainer_kwargs: Dict[str, Any] = {
            "model": training_mod.STATE.model,
            "reward_funcs": reward_func,
            "args": args,
            "train_dataset": dataset,
        }
        # Tip TRL: processing_class; older: tokenizer.
        try:
            trainer = GRPOTrainer(
                processing_class=training_mod.STATE.tokenizer,
                **trainer_kwargs,
            )
        except TypeError:
            trainer = GRPOTrainer(
                tokenizer=training_mod.STATE.tokenizer,
                **trainer_kwargs,
            )
        trainer.train()
    except Exception as e:
        return {
            "status": "error",
            "error": f"GRPOTrainer failed: {type(e).__name__}: {e}",
            "kind": "grpo",
            "trained": False,
        }

    training_mod.STATE.send_progress(100.0, "GRPO complete")
    return {
        "status": "ok",
        "kind": "grpo",
        "dry_run": False,
        "trained": True,
        "output_dir": output_dir,
        "n_prompts": len(prompts),
        "max_steps": max_steps,
        "model_name": model_name,
    }
