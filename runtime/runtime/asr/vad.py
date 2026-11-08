"""Silero-style ONNX voice activity for live turn-taking.

Scores PCM16 chunks and returns the index of the last speech sample.
Each live socket keeps its own recurrent state. 8 kHz and 16 kHz only.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

RATE_WINDOW = {
    8_000: 256,
    16_000: 512,
}
STATE_SHAPE = (2, 1, 128)


@dataclass
class SessionState:
    """Recurrent tensors for one live socket."""

    state: Any = None
    h: Any = None
    c: Any = None
    context: Optional[np.ndarray] = None
    # Samples that did not fill a full Silero window; prepended on the next score.
    leftover: Optional[np.ndarray] = None


@dataclass
class Ear:
    """One ONNX graph shared by many socket sessions."""

    model_path: str
    session: Any = None  # onnxruntime InferenceSession or a fake
    mode: str = ""  # stock | sequence
    input_width: int = 512
    context_samples: int = 0
    h_shape: Tuple[int, ...] = (1, 1, 64)
    c_shape: Tuple[int, ...] = (1, 1, 64)
    _sessions: Dict[str, SessionState] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self) -> None:
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if self.session is not None and self.mode:
            return
        if self.session is not None:
            # Injected fake or prebuilt session (tests).
            self._configure(self.session)
            return
        path = (self.model_path or "").strip()
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"neural VAD file missing: {path!r}")
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise ImportError(
                "onnxruntime is not installed (pip install onnxruntime)"
            ) from e
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._configure(self.session)

    def _configure(self, session: Any) -> None:
        inputs = {item.name: item for item in session.get_inputs()}
        outputs = {item.name for item in session.get_outputs()}
        if {"input", "state", "sr"}.issubset(inputs) and {"output", "stateN"}.issubset(
            outputs
        ):
            self.mode = "stock"
            shape = inputs["input"].shape
            width = int(shape[1]) if len(shape) > 1 and isinstance(shape[1], int) else 512
            if width == 576:
                self.input_width = 512
                self.context_samples = 64
            elif width in (256, 512):
                self.input_width = width
                self.context_samples = 0
            else:
                raise ValueError(f"unsupported stock input width {width}")
            return
        if {"input", "h", "c"}.issubset(inputs):
            self.mode = "sequence"
            shape = inputs["input"].shape
            if len(shape) > 1 and isinstance(shape[1], int):
                self.input_width = int(shape[1])
            else:
                self.input_width = 512
            self.context_samples = max(0, self.input_width - 512)
            self.h_shape = _static_shape(inputs["h"].shape, (1, 1, 64))
            self.c_shape = _static_shape(inputs["c"].shape, (1, 1, 64))
            return
        raise ValueError(
            "unsupported ONNX VAD graph; need input/state/sr or input/h/c"
        )

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def score(
        self,
        session_id: str,
        pcm: bytes,
        rate: int,
        threshold: float,
    ) -> int:
        """Return the last speech sample index, or -1 when the chunk is silence."""
        if rate not in RATE_WINDOW:
            raise ValueError(f"neural VAD rate must be 8000 or 16000, got {rate}")
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return -1
        window = RATE_WINDOW[rate]
        with self._lock:
            self._load_unlocked()
            st = self._sessions.get(session_id)
            if st is None:
                st = SessionState()
                self._sessions[session_id] = st
            if st.leftover is not None and st.leftover.size:
                samples = np.concatenate([st.leftover, samples])
            last = -1
            # Walk full windows. A short remainder is kept for the next packet.
            # Indices are relative to the concatenated leftover+pcm stream so the
            # caller still gets a last-speech index inside this request's pcm
            # when speech falls in the new samples; leftover-only speech maps
            # into the start of the new packet's index space via offset.
            left = 0 if st.leftover is None else int(st.leftover.size)
            pos = 0
            while pos + window <= samples.size:
                chunk = samples[pos : pos + window]
                prob, st = self._run_window(chunk, rate, st)
                if prob >= threshold:
                    # Map to index within this call's pcm (not the leftover prefix).
                    end = pos + window - 1
                    if end >= left:
                        last = end - left
                    elif last < 0:
                        # Speech only in leftover that just completed a window:
                        # treat as sample 0 of the new packet so silence clocks move.
                        last = 0
                pos += window
            if pos < samples.size:
                st.leftover = samples[pos:].copy()
            else:
                st.leftover = None
            self._sessions[session_id] = st
            return last

    def _run_window(
        self, chunk: np.ndarray, rate: int, st: SessionState
    ) -> Tuple[float, SessionState]:
        assert self.session is not None
        if self.mode == "stock":
            return self._run_stock(chunk, rate, st)
        return self._run_sequence(chunk, st)

    def _run_stock(
        self, chunk: np.ndarray, rate: int, st: SessionState
    ) -> Tuple[float, SessionState]:
        if st.state is None:
            st.state = np.zeros(STATE_SHAPE, dtype=np.float32)
        if self.context_samples > 0:
            ctx = st.context
            if ctx is None:
                ctx = np.zeros(self.context_samples, dtype=np.float32)
            framed = np.concatenate([ctx, chunk]).astype(np.float32)
            st.context = chunk[-self.context_samples :].copy()
        else:
            framed = chunk.astype(np.float32)
        feeds = {
            "input": framed.reshape(1, -1),
            "state": st.state,
            "sr": np.array(rate, dtype=np.int64),
        }
        out, state_n = self.session.run(["output", "stateN"], feeds)
        st.state = state_n
        return float(np.asarray(out).reshape(-1)[0]), st

    def _run_sequence(
        self, chunk: np.ndarray, st: SessionState
    ) -> Tuple[float, SessionState]:
        # Sequence graphs expect context+frame when width > 512.
        if self.context_samples > 0:
            ctx = st.context
            if ctx is None:
                ctx = np.zeros(self.context_samples, dtype=np.float32)
            need = self.input_width - self.context_samples
            frame = chunk[-need:] if chunk.size >= need else chunk
            if frame.size < need:
                frame = np.pad(frame, (0, need - frame.size))
            framed = np.concatenate([ctx, frame]).astype(np.float32)
            st.context = framed[-self.context_samples :].copy()
        else:
            framed = chunk.astype(np.float32)
            if framed.size < self.input_width:
                framed = np.pad(framed, (0, self.input_width - framed.size))
            elif framed.size > self.input_width:
                framed = framed[-self.input_width :]
        if st.h is None or st.c is None:
            st.h = np.zeros(self.h_shape, dtype=np.float32)
            st.c = np.zeros(self.c_shape, dtype=np.float32)
        feeds = {
            "input": framed.reshape(1, -1),
            "h": st.h,
            "c": st.c,
        }
        outs = self.session.run(None, feeds)
        # Prefer named outputs when present.
        names = [o.name for o in self.session.get_outputs()]
        by_name = dict(zip(names, outs))
        prob = by_name.get("speech_probs", by_name.get("output", outs[0]))
        if "hn" in by_name:
            st.h = by_name["hn"]
        elif len(outs) > 1:
            st.h = outs[1]
        if "cn" in by_name:
            st.c = by_name["cn"]
        elif len(outs) > 2:
            st.c = outs[2]
        return float(np.asarray(prob).reshape(-1)[0]), st


def _static_shape(shape: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    out = []
    for i, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        elif i < len(default):
            out.append(default[i])
        else:
            out.append(1)
    return tuple(out) if out else default
