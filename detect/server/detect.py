"""OWLv2 detect helpers.

The HTTP server lives in app.py. This module stays importable without torch:
a missing model directory answers status=wire_only instead of inventing boxes.
Weights are a local directory. A Hub repo id is never passed to from_pretrained.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional


def model_dir_ready(path: str) -> bool:
    path = (path or "").strip()
    return bool(path) and os.path.isdir(path)


def image_b64(body: dict[str, Any]) -> str:
    for key in ("image", "image_base64", "frame"):
        raw = body.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    url = body.get("image_url")
    if isinstance(url, str) and url.strip().startswith("data:"):
        return url.strip()
    return ""


def decode_image(raw: str) -> bytes:
    text = raw.strip()
    if text.startswith("data:"):
        comma = text.find(",")
        if comma < 0:
            raise ValueError("data URL has no payload")
        text = text[comma + 1 :]
    try:
        return base64.b64decode(text, validate=False)
    except Exception as e:
        raise ValueError("image is not base64") from e


def queries_of(body: dict[str, Any]) -> list[str]:
    raw = body.get("queries")
    if raw is None:
        raw = body.get("labels")
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        out = [str(item).strip() for item in raw if str(item).strip()]
        if out:
            return out
    text = body.get("text")
    if isinstance(text, str) and text.strip():
        return [text.strip()]
    return []


def threshold_of(body: dict[str, Any]) -> float:
    raw = body.get("threshold", 0.1)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.1
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def wire_only(reason: str) -> dict[str, Any]:
    return {
        "status": "wire_only",
        "model": "owlv2",
        "detections": [],
        "reason": reason,
    }


def boxes_from_results(results: dict[str, Any], queries: list[str]) -> list[dict[str, Any]]:
    """Turn one post_process_object_detection result into pixel boxes.

    box is [x1, y1, x2, y2] in the decoded image, origin at the top left.
    """
    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels")
    if boxes is None or scores is None or labels is None:
        return []
    out: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        coords = [round(float(v), 2) for v in list(box)]
        idx = int(label)
        name = queries[idx] if 0 <= idx < len(queries) else str(idx)
        out.append({"label": name, "score": round(float(score), 4), "box": coords})
    return out


class Detector:
    """One OWLv2 model. load() is a no-op until a local directory exists."""

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        self.model_dir = (model_dir or "").strip()
        self.device = (device or "cpu").strip() or "cpu"
        self._model: Any = None
        self._processor: Any = None
        self.reason = ""

    def load(self) -> None:
        if self._model is not None:
            return
        if not model_dir_ready(self.model_dir):
            self.reason = "OWLv2 weights directory is unset or missing"
            return
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as e:
            self.reason = f"owlv2 runtime missing: {e}"
            return
        self._processor = Owlv2Processor.from_pretrained(self.model_dir, local_files_only=True)
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self._model = Owlv2ForObjectDetection.from_pretrained(
            self.model_dir, local_files_only=True, dtype=dtype
        )
        self._model.to(self.device)
        self._model.eval()
        self.reason = ""

    def detect(self, body: dict[str, Any]) -> dict[str, Any]:
        self.load()
        raw = image_b64(body)
        if not raw:
            raise ValueError("image, image_base64, image_url, or frame is required")
        if self._model is None:
            return wire_only(self.reason or "OWLv2 is not loaded")
        queries = queries_of(body)
        if not queries:
            raise ValueError("queries, labels, or text is required")
        return self._run(decode_image(raw), queries, threshold_of(body))

    def _run(self, image_bytes: bytes, queries: list[str], threshold: float) -> dict[str, Any]:
        import io

        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(text=[queries], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        target = torch.tensor([[image.height, image.width]], device=self.device)
        processed = self._processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target
        )[0]
        moved = {k: v.detach().cpu() for k, v in processed.items()}
        return {
            "status": "ok",
            "model": "owlv2",
            "detections": boxes_from_results(moved, queries),
            "image": {"width": image.width, "height": image.height},
        }


def detector_from_env() -> Detector:
    return Detector(
        os.environ.get("ZEROLLAMA_OWL_MODEL", ""),
        os.environ.get("ZEROLLAMA_DETECT_DEVICE", "cpu"),
    )
