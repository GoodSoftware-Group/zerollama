"""T10 / T10b — GRPO / RL job kind on the stock training path (TRL).

WHY one backend: Unsloth's GRPO/Studio path is a second stack (and Studio is AGPL).
Zerollama keeps PEFT + Transformers/``trl`` on the same embed JOB_QUEUE as SFT so T6
idle-wait / defer and the VRAM broker stay one policy.

WHY dry-run by default: CI and hosts without ``trl`` still validate payloads; operators
must opt into ``grpo_run=true`` on a free GPU lab.

WHY HTTP rewards are named (not callables): JSON cannot carry Python callables.
T10b expands builtins + ``reward_url`` (agent hook) + ``reward_hf`` (HF classifier).
In-process callers of ``process_grpo_request`` may still pass a real callable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

RewardFunc = Callable[..., List[float]]

# HTTP-safe builtin names → canonical.
_BUILTIN_ALIASES: Dict[str, str] = {
    "length": "length",
    "len": "length",
    "nonempty": "nonempty",
    "format_json": "format_json",
    "json": "format_json",
    "exact_match": "exact_match",
    "em": "exact_match",
}
_KNOWN_BUILTINS = frozenset(_BUILTIN_ALIASES.keys())


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


def normalize_grpo_references(data: Dict[str, Any]) -> List[str]:
    """Parallel references for exact_match: references[] or training_data answer fields."""
    raw = data.get("references")
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if item is None:
                out.append("")
            else:
                out.append(str(item))
        return out
    rows = data.get("training_data")
    if not isinstance(rows, list):
        return []
    out = []
    for item in rows:
        if isinstance(item, dict):
            ref = (
                item.get("answer")
                or item.get("reference")
                or item.get("label")
                or item.get("response")
                or item.get("expected")
            )
            out.append("" if ref is None else str(ref))
        else:
            out.append("")
    return out


def _truthy_flag(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return False


def _looks_like_hf_id(s: str) -> bool:
    """org/name (or nested) — not a builtin alias."""
    t = s.strip()
    if not t or t.lower() in _KNOWN_BUILTINS:
        return False
    if "/" not in t:
        return False
    if t.startswith(("http://", "https://", "file:", "/")):
        return False
    return True


def _reward_url(data: Dict[str, Any]) -> Optional[str]:
    for key in ("reward_url", "reward_hook"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _validate_reward_url(url: str) -> Optional[str]:
    """Return error string or None if ok.

    WHY http(s) only: file:// and other schemes are accidental SSRF / local-read vectors
    for a job payload that agents may fill. Operators point at their own lab hook.
    """
    try:
        p = urlparse(url)
    except Exception:
        return f"reward_url unparseable: {url!r}"
    if p.scheme not in ("http", "https"):
        return "reward_url must be http(s) (file:// and other schemes rejected)"
    if not p.netloc:
        return "reward_url missing host"
    return None


def _hf_model_id(data: Dict[str, Any]) -> Optional[str]:
    for key in ("reward_hf", "reward_model"):
        v = data.get(key)
        if isinstance(v, str) and _looks_like_hf_id(v):
            return v.strip()
    return None


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
            # reward_model may be an HF id — skip here.
            if key == "reward_model" and _looks_like_hf_id(v):
                continue
            if name in _BUILTIN_ALIASES:
                return _BUILTIN_ALIASES[name]
    return None


def length_reward(completions: List[str], **kwargs: Any) -> List[float]:
    """Smoke-friendly builtin: reward = completion character length."""
    return [float(len(c) if c is not None else 0) for c in completions]


def nonempty_reward(completions: List[str], **kwargs: Any) -> List[float]:
    return [1.0 if (c or "").strip() else 0.0 for c in completions]


def format_json_reward(completions: List[str], **kwargs: Any) -> List[float]:
    out: List[float] = []
    for c in completions:
        try:
            json.loads(c if c is not None else "")
            out.append(1.0)
        except Exception:
            out.append(0.0)
    return out


def make_exact_match_reward(references: List[str]) -> RewardFunc:
    def exact_match_reward(completions: List[str], **kwargs: Any) -> List[float]:
        scores: List[float] = []
        for i, c in enumerate(completions):
            ref = references[i] if i < len(references) else ""
            scores.append(
                1.0 if (c or "").strip() == (ref or "").strip() else 0.0
            )
        return scores

    return exact_match_reward


def make_url_reward(url: str, timeout_s: float = 5.0) -> RewardFunc:
    """POST completions to an agent hook; expect {\"rewards\": [float, ...]}."""

    def url_reward(completions: List[str], **kwargs: Any) -> List[float]:
        prompts = kwargs.get("prompts") or kwargs.get("prompt")
        body: Dict[str, Any] = {"completions": list(completions)}
        if prompts is not None:
            body["prompts"] = prompts
        raw = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=raw,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"reward_url HTTP {e.code}: {e.reason}") from e
        except Exception as e:
            raise RuntimeError(f"reward_url failed: {type(e).__name__}: {e}") from e
        rewards = payload.get("rewards") if isinstance(payload, dict) else None
        if not isinstance(rewards, list) or len(rewards) != len(completions):
            raise RuntimeError(
                "reward_url must return JSON {\"rewards\": [float, ...]} "
                f"with len=={len(completions)}"
            )
        return [float(x) for x in rewards]

    return url_reward


def make_hf_reward(model_id: str) -> RewardFunc:
    """Lazy-load HF sequence-classification reward model (lab / grpo_run)."""

    state: Dict[str, Any] = {"tok": None, "model": None}

    def hf_reward(completions: List[str], **kwargs: Any) -> List[float]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if state["model"] is None:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.eval()
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            state["tok"] = tok
            state["model"] = model
        tok = state["tok"]
        model = state["model"]
        device = next(model.parameters()).device
        enc = tok(
            list(completions),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        # Single-logit or take first class score — common RM layout.
        if logits.ndim == 2 and logits.shape[-1] == 1:
            scores = logits.squeeze(-1)
        else:
            scores = logits[..., 0]
        return [float(x) for x in scores.detach().cpu().tolist()]

    return hf_reward


def _make_builtin(name: str, data: Dict[str, Any]) -> Tuple[Optional[RewardFunc], Optional[str]]:
    if name == "length":
        return length_reward, None
    if name == "nonempty":
        return nonempty_reward, None
    if name == "format_json":
        return format_json_reward, None
    if name == "exact_match":
        refs = normalize_grpo_references(data)
        prompts = normalize_grpo_prompts(data)
        if not refs:
            return None, (
                "exact_match requires references[] or training_data "
                "answer/reference/label/response fields"
            )
        if len(refs) != len(prompts):
            return None, (
                f"exact_match: len(references)={len(refs)} != len(prompts)={len(prompts)}"
            )
        return make_exact_match_reward(refs), None
    return None, f"unknown reward_builtin {name!r}"


def resolve_reward_func(data: Dict[str, Any]) -> Tuple[Optional[RewardFunc], Optional[str]]:
    """Return (func, error). HTTP path: builtins / url / hf. In-process: callables too."""
    for key in ("reward_funcs", "reward_fn", "reward"):
        v = data.get(key)
        if callable(v):
            return v, None  # type: ignore[return-value]

    url = _reward_url(data)
    if url:
        err = _validate_reward_url(url)
        if err:
            return None, err
        timeout = float(data.get("reward_url_timeout", 5.0) or 5.0)
        return make_url_reward(url, timeout_s=timeout), None

    name = _builtin_name(data)
    if name:
        return _make_builtin(name, data)

    hf_id = _hf_model_id(data)
    if hf_id:
        return make_hf_reward(hf_id), None

    # Explicit reject of JSON-only "callable" pretenders.
    for key in ("reward_funcs", "reward_fn"):
        if data.get(key) is not None and not callable(data.get(key)):
            return None, (
                f"{key} cannot be sent over JSON; use reward_builtin "
                "(length|nonempty|format_json|exact_match), reward_url, or reward_hf"
            )

    v = data.get("reward_builtin")
    if isinstance(v, str) and v.strip() and v.strip().lower() not in _KNOWN_BUILTINS:
        return None, (
            f"unknown reward_builtin {v!r} "
            "(want length|nonempty|format_json|exact_match)"
        )

    rm = data.get("reward_model")
    if isinstance(rm, str) and rm.strip():
        low = rm.strip().lower()
        if low not in _KNOWN_BUILTINS and not _looks_like_hf_id(rm):
            return None, (
                "reward_model must be a builtin name or HF id (org/name); "
                "use reward_hf for clarity"
            )

    if isinstance(data.get("reward"), str) and data["reward"].strip():
        low = data["reward"].strip().lower()
        if low not in _KNOWN_BUILTINS:
            return None, (
                "reward string must be a builtin name "
                "(length|nonempty|format_json|exact_match)"
            )

    return None, (
        "reward signal required: reward_builtin=length|nonempty|format_json|exact_match, "
        "reward_url=http(s)://…, reward_hf=org/name, "
        "or a Python callable via reward_funcs (in-process)"
    )


def describe_reward(data: Dict[str, Any]) -> str:
    """Short label for dry-run / logs."""
    if any(callable(data.get(k)) for k in ("reward_funcs", "reward_fn", "reward")):
        return "callable"
    if _reward_url(data):
        return "url"
    name = _builtin_name(data)
    if name:
        return name
    hf = _hf_model_id(data)
    if hf:
        return f"hf:{hf}"
    return "unknown"


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
            "pip install 'trl>=0.14,<1' (same interpreter as training.py) "
            "or re-sync requirements-training.txt"
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
                "upgrade trl (pip install -U 'trl>=0.14,<1')"
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
    reward_label = describe_reward(request)

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
            "reward": reward_label,
            "reward_builtin": _builtin_name(request) or reward_label,
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


def _shield_sibling_mlx_false_positive() -> None:
    """Neutralize empty sibling ``mlx/`` namespace that fools transformers.

    WHY: Mac lab often has repo-root or ``../mlx`` checkout on ``sys.path``. That
    makes ``importlib`` report package ``mlx`` without ``mlx.core``. Transformers
    then sets ``_is_mlx_available=True`` and ``is_tensor`` crashes mid-GRPO step.
    Real Apple MLX installs provide ``mlx.core`` — we leave those alone.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("mlx.core") is not None:
        return
    # Drop path entries that are the sibling mlx tree itself (keep "" / "." for repo imports).
    cleaned: List[str] = []
    for p in sys.path:
        norm = p.rstrip("/")
        if norm.endswith("/mlx") and not norm.endswith("/site-packages/mlx"):
            continue
        cleaned.append(p)
    sys.path[:] = cleaned
    sys.modules.pop("mlx", None)
    try:
        import transformers.utils.generic as tg

        tg._is_mlx_available = False
    except Exception:
        pass
    try:
        import transformers.utils.import_utils as iu

        if hasattr(iu.is_mlx_available, "cache_clear"):
            iu.is_mlx_available.cache_clear()
    except Exception:
        pass


def _run_grpo_trainer(request: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal GRPOTrainer wire — lab only (grpo_run=true).

    WHY caller owns ``training_active``: clearing it here opened a window where
    OccupiesGPU / block-inference saw idle before the worker finished the job.
    """
    import training as training_mod

    _shield_sibling_mlx_false_positive()
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
    # TRL tip (0.29): generation_batch_size must be divisible by num_generations.
    # WHY lab defaults 2/2: tip default is 8/8 — too heavy for tiny smoke models / single GPU.
    num_generations = int(request.get("num_generations", 2) or 2)
    num_generations = max(1, num_generations)
    batch_size = int(request.get("batch_size", num_generations) or num_generations)
    batch_size = max(1, batch_size)
    gen_batch = int(request.get("generation_batch_size", 0) or 0)
    if gen_batch <= 0:
        gen_batch = batch_size if batch_size % num_generations == 0 else num_generations * batch_size
    if gen_batch % num_generations != 0:
        return {
            "status": "error",
            "error": (
                f"generation_batch_size ({gen_batch}) must be divisible by "
                f"num_generations ({num_generations})"
            ),
            "kind": "grpo",
            "trained": False,
        }
    try:
        args = GRPOConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            num_generations=num_generations,
            generation_batch_size=gen_batch,
            logging_steps=1,
            report_to=[],
        )
    except TypeError as e:
        # Older trl: may lack generation_batch_size — retry minimal knobs.
        try:
            args = GRPOConfig(
                output_dir=output_dir,
                max_steps=max_steps,
                per_device_train_batch_size=batch_size,
                num_generations=num_generations,
                logging_steps=1,
                report_to=[],
            )
        except TypeError as e2:
            return {
                "status": "error",
                "error": f"GRPOConfig API mismatch: {e}; retry: {e2}",
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
        "reward": describe_reward(request),
    }
