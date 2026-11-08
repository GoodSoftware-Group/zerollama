"""Local Kokoro ONNX speaker for live and file speech.

Loads once. One lock covers phonemize plus ONNX so two calls do not share
espeak-ng or the session. Sample rate is 24000. Float samples become int16.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import threading
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

SAMPLE_RATE = 24_000
DEFAULT_VOICE = "af_heart"

# Kokoro v1.1 prefixes. This ONNX tokenizer only phonemizes en-us and en-gb.
_NON_ENGLISH_PREFIXES = (
    "ef_",
    "em_",
    "ff_",
    "fm_",
    "hf_",
    "hm_",
    "if_",
    "im_",
    "jf_",
    "jm_",
    "pf_",
    "pm_",
    "zf_",
    "zm_",
)


def english_voice(voice: str) -> bool:
    """False for Kokoro's non-English id prefixes."""
    v = (voice or "").strip().lower()
    if not v:
        return False
    return not any(v.startswith(p) for p in _NON_ENGLISH_PREFIXES)


def lang_for_voice(voice: str) -> str:
    """bf_/bm_ voices are UK English; every other English id is US English."""
    v = (voice or "").strip().lower()
    if v.startswith("bf_") or v.startswith("bm_"):
        return "en-gb"
    return "en-us"


def resolve_lang(voice: str, lang: str = "") -> str:
    """Pick en-us or en-gb. An explicit English tag overrides the voice prefix."""
    voice = (voice or "").strip()
    if voice and not english_voice(voice):
        raise ValueError(f"kokoro speaks English only, voice {voice!r}")
    raw = (lang or "").strip().lower().replace("_", "-")
    if raw == "":
        return lang_for_voice(voice)
    if raw in ("en", "en-us"):
        return "en-us"
    if raw in ("en-gb", "en-uk"):
        return "en-gb"
    raise ValueError("kokoro speaks English only (en-us or en-gb)")


def float_to_pcm16(samples: Any) -> bytes:
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# The ONNX graph has no emotion head. These words scale the requested speed.
_EMOTION_SCALE = {
    "calm": 0.85,
    "sad": 0.80,
    "serious": 0.90,
    "neutral": 1.00,
    "happy": 1.10,
    "excited": 1.20,
    "angry": 1.25,
}


def apply_emotion(speed: float, emotion: str = "") -> Tuple[float, str]:
    """Scale speed once. An omitted emotion leaves speed unchanged."""
    word = (emotion or "").strip().lower()
    if not word:
        return float(speed), ""
    scale = _EMOTION_SCALE.get(word)
    if scale is None:
        names = ", ".join(sorted(_EMOTION_SCALE))
        raise ValueError(f"kokoro emotion must be one of: {names}")
    out = float(speed) * scale
    if not 0.5 <= out <= 2.0:
        raise ValueError("kokoro emotion pushes speed outside 0.5 to 2")
    return out, word


def parse_pauses(value: Any) -> dict[str, float]:
    """Canonical phoneme_silence: a map, or a list of [character, seconds]."""
    if value is None or value == "" or value == {} or value == []:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"phoneme_silence: {e}") from e
    pairs: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        pairs = [(str(k), v) for k, v in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError("phoneme_silence pairs must be [character, seconds]")
            pairs.append((str(item[0]), item[1]))
    else:
        raise ValueError("phoneme_silence must be a map")
    if len(pairs) > 32:
        raise ValueError("phoneme_silence accepts at most 32 phonemes")
    out: dict[str, float] = {}
    for key, raw in pairs:
        if len(key) != 1:
            raise ValueError("phoneme_silence keys must be a single character")
        try:
            seconds = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError("phoneme_silence must be between 0 and 5") from e
        if math.isnan(seconds) or math.isinf(seconds) or seconds < 0 or seconds > 5:
            raise ValueError("phoneme_silence must be between 0 and 5")
        out[key] = seconds
    return out


def split_pauses(text: str, pauses: Mapping[str, float]) -> list[tuple[str, float]]:
    """Split text on marked characters. Each piece is followed by its silence."""
    if not pauses:
        return [(text, 0.0)]
    pieces: list[tuple[str, float]] = []
    buf: list[str] = []
    for ch in text:
        if ch in pauses:
            pieces.append(("".join(buf), float(pauses[ch])))
            buf = []
            continue
        buf.append(ch)
    if buf or not pieces:
        pieces.append(("".join(buf), 0.0))
    return pieces


def silence_pcm(rate: int, seconds: float) -> bytes:
    n = int(round(int(rate) * float(seconds)))
    if n <= 0:
        return b""
    return b"\x00\x00" * n


def pcm16_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm)
    return buf.getvalue()


@dataclass
class Speaker:
    """One Kokoro graph shared by every speak call."""

    model_path: str
    voices_path: str
    kokoro: Any = None  # kokoro_onnx.Kokoro or a fake
    rate: int = SAMPLE_RATE
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self) -> None:
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if self.kokoro is not None:
            return
        model = (self.model_path or "").strip()
        voices = (self.voices_path or "").strip()
        if not model or not os.path.isfile(model):
            raise FileNotFoundError(f"kokoro model missing: {model!r}")
        if not voices or not os.path.isfile(voices):
            raise FileNotFoundError(f"kokoro voices missing: {voices!r}")
        try:
            from kokoro_onnx import Kokoro  # type: ignore
        except ImportError as e:
            raise ImportError(
                "kokoro-onnx is not installed (pip install kokoro-onnx)"
            ) from e
        self.kokoro = Kokoro(model, voices)
        self.rate = SAMPLE_RATE

    def speak(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        lang: str = "",
        phoneme_silence: Any = None,
        emotion: str = "",
    ) -> Tuple[int, bytes]:
        """Return (rate, 16-bit mono WAV bytes)."""
        voice, lang, speed, pieces = self._prepare(
            text, voice, speed, lang, phoneme_silence, emotion
        )
        parts: list[bytes] = []
        rate = SAMPLE_RATE
        with self._lock:
            self._load_unlocked()
            assert self.kokoro is not None
            for piece, seconds in pieces:
                if piece.strip():
                    samples, got = self.kokoro.create(
                        piece, voice=voice, speed=speed, lang=lang
                    )
                    rate = int(got) or rate
                    parts.append(float_to_pcm16(samples))
                if seconds > 0:
                    parts.append(silence_pcm(rate, seconds))
        return rate, pcm16_wav(b"".join(parts), rate)

    def speak_stream(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        lang: str = "",
        phoneme_silence: Any = None,
        emotion: str = "",
        emit: Optional[Callable[[int, bytes], None]] = None,
    ) -> None:
        """Run create_stream and call emit(rate, pcm16) per phoneme batch."""
        if emit is None:
            raise ValueError("emit required")
        voice, lang, speed, pieces = self._prepare(
            text, voice, speed, lang, phoneme_silence, emotion
        )

        async def _run() -> None:
            # Hold the lock for the whole utterance: espeak-ng is process-global
            # and two ONNX runs must not overlap.
            with self._lock:
                self._load_unlocked()
                assert self.kokoro is not None
                rate = SAMPLE_RATE
                for piece, seconds in pieces:
                    if piece.strip():
                        stream = self.kokoro.create_stream(
                            piece, voice=voice, speed=speed, lang=lang
                        )
                        async for samples, got in stream:
                            rate = int(got) or rate
                            emit(rate, float_to_pcm16(samples))
                    if seconds > 0:
                        pcm = silence_pcm(rate, seconds)
                        if pcm:
                            emit(rate, pcm)

        asyncio.run(_run())

    def _prepare(
        self,
        text: str,
        voice: str,
        speed: float,
        lang: str,
        phoneme_silence: Any,
        emotion: str,
    ) -> Tuple[str, str, float, list[tuple[str, float]]]:
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")
        if not 0.5 <= float(speed) <= 2.0:
            raise ValueError(f"speed must be between 0.5 and 2, got {speed}")
        voice = (voice or "").strip() or DEFAULT_VOICE
        lang = resolve_lang(voice, lang)
        speed, _word = apply_emotion(speed, emotion)
        pieces = split_pauses(text, parse_pauses(phoneme_silence))
        return voice, lang, speed, pieces
