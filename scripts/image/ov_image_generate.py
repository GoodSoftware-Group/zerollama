#!/usr/bin/env python3
"""OpenVINO GenAI text-to-image for zerollama external-image hook."""
from __future__ import annotations

import os
import sys


def main() -> int:
    model_dir = os.environ.get("OLLAMA_OV_MODEL_DIR", "")
    output = os.environ.get("OLLAMA_IMAGE_OUTPUT", "")
    prompt = os.environ.get("OLLAMA_IMAGE_PROMPT", "")
    if not model_dir or not os.path.isdir(model_dir):
        print("OLLAMA_OV_MODEL_DIR must point to an OpenVINO IR model directory", file=sys.stderr)
        return 1
    if not output:
        print("OLLAMA_IMAGE_OUTPUT is required", file=sys.stderr)
        return 1
    if not prompt:
        print("OLLAMA_IMAGE_PROMPT is empty", file=sys.stderr)
        return 1

    device = os.environ.get("OLLAMA_OV_DEVICE", "GPU")
    width = int(os.environ.get("OLLAMA_IMAGE_WIDTH", os.environ.get("OLLAMA_OV_DEFAULT_WIDTH", "512")))
    height = int(os.environ.get("OLLAMA_IMAGE_HEIGHT", os.environ.get("OLLAMA_OV_DEFAULT_HEIGHT", "512")))
    steps = int(os.environ.get("OLLAMA_OV_STEPS", "20"))
    cfg = float(os.environ.get("OLLAMA_OV_CFG_SCALE", "7.5"))
    seed_raw = os.environ.get("OLLAMA_IMAGE_SEED", "-1")

    try:
        import openvino_genai as ov_genai
        from PIL import Image
    except ImportError as e:
        print(f"openvino_genai not installed: {e}", file=sys.stderr)
        return 1

    pipe = ov_genai.Text2ImagePipeline(model_dir, device)
    kwargs: dict = {
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "num_images_per_prompt": 1,
    }
    if seed_raw not in ("", "-1"):
        try:
            kwargs["generator_seed"] = int(seed_raw)
        except ValueError:
            pass

    image_tensor = pipe.generate(prompt, **kwargs)
    image = Image.fromarray(image_tensor.data[0])
    image.save(output, format="PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
