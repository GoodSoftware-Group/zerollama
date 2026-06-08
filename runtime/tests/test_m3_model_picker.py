"""Tests for M3 / Metal sign-off model picker (manifest → text GGUF)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Inline copy of smoke_m3_pick_text_gguf logic for unit testing (bash heredoc mirror).
def pick_smallest_text_gguf(manifests_root: Path, blobs_root: Path) -> tuple[str, str] | None:
    best: tuple[int, str, str] | None = None
    for mf in sorted(manifests_root.rglob("latest")):
        try:
            m = json.loads(mf.read_text())
            if any("projector" in (layer.get("mediaType") or "") for layer in m.get("layers", [])):
                continue
            cfg_path = blobs_root / m["config"]["digest"].replace("sha256:", "sha256-")
            cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
            fam = (cfg.get("model_family") or "").lower()
            if fam in ("nomic-bert", "bert", "embed"):
                continue
            if "gemma" in fam and cfg.get("model_type") not in (None, "", "llama"):
                continue
            for layer in m.get("layers", []):
                if layer.get("mediaType") != "application/vnd.ollama.image.model":
                    continue
                d = layer["digest"].replace("sha256:", "sha256-")
                path = blobs_root / d
                size = int(layer.get("size") or 0)
                if not path.is_file():
                    continue
                if best is None or size < best[0]:
                    best = (size, str(path), mf.parent.name)
                break
        except Exception:
            pass
    if best:
        return best[1], best[2]
    return None


def _write_manifest(
    root: Path,
    name: str,
    *,
    blob_name: str,
    blob_size: int,
    family: str = "llama",
    extra_layers: list[dict] | None = None,
    model_type: str | None = "llama",
) -> Path:
    blobs = root / "blobs"
    manifests = root / "manifests/registry.ollama.ai/library" / name
    manifests.mkdir(parents=True, exist_ok=True)
    blob_path = blobs / blob_name
    blob_path.write_bytes(b"x" * min(blob_size, 64))
    cfg_digest = f"sha256-{name}-cfg"
    (blobs / cfg_digest).write_text(
        json.dumps({"model_family": family, "model_type": model_type})
    )
    layers = list(extra_layers or [])
    layers.append(
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": f"sha256:{blob_name.removeprefix('sha256-')}",
            "size": blob_size,
        }
    )
    layers.insert(
        0,
        {"mediaType": "application/vnd.ollama.image.config", "digest": f"sha256:{name}-cfg"},
    )
    (manifests / "latest").write_text(
        json.dumps({"config": {"digest": f"sha256:{name}-cfg"}, "layers": layers})
    )
    return blob_path


def test_picker_skips_projector_and_embed(tmp_path: Path):
    root = tmp_path / "models"
    blobs = root / "blobs"
    manifests = root / "manifests/registry.ollama.ai/library"
    blobs.mkdir(parents=True)

    _write_manifest(
        root,
        "tiny-text",
        blob_name="sha256-aaa",
        blob_size=100,
        family="qwen3",
    )
    _write_manifest(
        root,
        "vision-model",
        blob_name="sha256-bbb",
        blob_size=50,
        family="gemma3",
        extra_layers=[{"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:p"}],
        model_type="clip",
    )
    _write_manifest(
        root,
        "embed-model",
        blob_name="sha256-ccc",
        blob_size=10,
        family="nomic-bert",
    )

    result = pick_smallest_text_gguf(manifests, blobs)
    assert result is not None
    path, tag = result
    assert tag == "tiny-text"
    assert "sha256-aaa" in path


def test_picker_prefers_smallest_text(tmp_path: Path):
    root = tmp_path / "models"
    blobs = root / "blobs"
    manifests = root / "manifests/registry.ollama.ai/library"
    blobs.mkdir(parents=True)

    _write_manifest(root, "big", blob_name="sha256-big", blob_size=9000, family="llama")
    _write_manifest(root, "small", blob_name="sha256-small", blob_size=500, family="llama")

    result = pick_smallest_text_gguf(manifests, blobs)
    assert result is not None
    _, tag = result
    assert tag == "small"
