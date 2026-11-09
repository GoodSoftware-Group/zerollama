"""Resident Qwen3-ASR ear.

Loads ``qwen_asr.Qwen3ASRModel`` once from a local directory. A Hub id is
never passed to ``from_pretrained``. Missing package or directory means the
caller starts Whisper instead.
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, Optional


def model_dir_ready(path: str) -> bool:
    """True when path is an existing directory (never a Hub repo id)."""
    path = (path or "").strip()
    if not path:
        return False
    return os.path.isdir(path)


class Ear:
    """One Qwen3-ASR model for many clips."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        model: Any = None,
    ) -> None:
        self.model_dir = (model_dir or "").strip()
        self.device = (device or "cpu").strip() or "cpu"
        self._model = model
        self._lock = threading.Lock()

    def load(self) -> None:
        """Import and load weights. Raises when the package or directory is missing."""
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if self._model is not None:
            return
        if not model_dir_ready(self.model_dir):
            raise FileNotFoundError(
                f"qwen3-asr model directory missing: {self.model_dir!r}"
            )
        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "qwen-asr is not installed (pip install qwen-asr)"
            ) from e

        kwargs: dict[str, Any] = {
            "max_inference_batch_size": 1,
            "max_new_tokens": 1024,
            "device_map": self.device,
        }
        if self.device == "cpu":
            kwargs["dtype"] = _torch_float32()
        else:
            kwargs["dtype"] = _torch_bfloat16()

        self._model = Qwen3ASRModel.from_pretrained(self.model_dir, **kwargs)

    def transcribe(self, wav: bytes, language: Optional[str] = None) -> str:
        """Transcribe one WAV.

        The first pass leaves language unset. Qwen then returns empty on
        silence instead of Okay / Hello / a period. A short English "Hi"
        can still come back as Japanese はい。 When the caller asked for
        English and the text has no Latin letters, a second pass forces
        English.
        """
        with self._lock:
            self._load_unlocked()
            assert self._model is not None
            path = _write_wav(wav)
            try:
                text = _heard_words(self._infer(path, None))
                lang = (language or "").strip()
                if text and _english_requested(lang) and not _has_latin(text):
                    forced = _heard_words(self._infer(path, lang))
                    if forced:
                        text = forced
                return text
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _infer(self, path: str, language: Optional[str]) -> str:
        assert self._model is not None
        results = self._model.transcribe(
            audio=path,
            language=language,
            return_time_stamps=False,
        )
        if not results:
            return ""
        text = getattr(results[0], "text", None)
        if text is None and isinstance(results[0], dict):
            text = results[0].get("text")
        return text or ""


def _heard_words(text: Optional[str]) -> str:
    """Drop a transcript that is only punctuation or whitespace."""
    text = (text or "").strip()
    if any(ch.isalnum() for ch in text):
        return text
    return ""


def _english_requested(language: str) -> bool:
    return language.lower() in {"en", "english"}


def _has_latin(text: str) -> bool:
    return any(ch.isascii() and ch.isalpha() for ch in text)


def _torch_float32() -> Any:
    import torch

    return torch.float32


def _torch_bfloat16() -> Any:
    import torch

    return torch.bfloat16


def _write_wav(wav: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="qwen3-asr-")
    try:
        os.write(fd, wav)
    finally:
        os.close(fd)
    return path
