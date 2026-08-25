#!/usr/bin/env python3
"""Dump Wan Python T5 + noise (+ optional DiT step-0 pred) for parity with wan-c.

When WAN_REPO + ckpt exist, loads UMT5 encoder and writes:
  meta.json, t5_emb.npy [seq,4096], noise_latent.npy [16,T,H,W]

With --dit, also loads WanModel and runs one forward (CFG=1 by default)
at the first UniPC timestep. Use --noise-from to inject wan-c noise.f32
so DiT A/B is not confounded by RNG.

Usage:
  WAN_REPO=~/.zerollama/third_party/wan/Wan2.1 \\
  WAN_CKPT=~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B \\
    python3 tools/parity_dump.py --prompt "a red apple" --out dumps/parity_py \\
      --width 64 --height 64 --frames 5 --seed 42

  # DiT step-0 A/B (inject C noise):
  python3 tools/parity_dump.py --prompt "a red apple on a wooden table" \\
      --out dumps/parity_py --dit --noise-from dumps/parity_c \\
      --width 64 --height 64 --frames 5 --steps 1 --cfg 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def write_stub(out: Path, prompt: str, steps: int, ckpt: Path, shape: tuple) -> None:
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    c, t, h, w = shape
    meta = {
        "prompt": prompt,
        "ckpt_dir": str(ckpt),
        "steps": steps,
        "mode": "stub",
        "latent_shape": [c, t, h, w],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    rng = np.random.RandomState(0)
    np.save(out / "noise_latent.npy", rng.randn(c, t, h, w).astype(np.float32))
    np.save(out / "t5_emb_stub.npy", rng.randn(512, 4096).astype(np.float32))
    print(f"wrote stub dumps under {out}")


def _load_noise_from_c(noise_from: Path, shape: tuple) -> "np.ndarray":
    import numpy as np

    path = noise_from / "noise.f32"
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.fromfile(path, dtype=np.float32)
    expect = int(np.prod(shape))
    if arr.size != expect:
        raise ValueError(f"noise.f32 size {arr.size} != {shape} ({expect})")
    return arr.reshape(shape)


def _load_t5_from_c(noise_from: Path) -> "np.ndarray | None":
    import numpy as np

    meta_p = noise_from / "meta.json"
    t5_p = noise_from / "t5_emb.f32"
    if not meta_p.is_file() or not t5_p.is_file():
        return None
    meta = json.loads(meta_p.read_text())
    shape = tuple(meta.get("t5_shape", []))
    if len(shape) != 2:
        return None
    arr = np.fromfile(t5_p, dtype=np.float32)
    if arr.size != int(np.prod(shape)):
        return None
    return arr.reshape(shape)


def try_wan_dump(
    out: Path,
    prompt: str,
    steps: int,
    ckpt: Path,
    repo: Path,
    width: int,
    height: int,
    frames: int,
    seed: int,
    *,
    dit: bool = False,
    cfg: float = 1.0,
    shift: float = 5.0,
    noise_from: Path | None = None,
    t5_from_c: bool = False,
) -> bool:
    try:
        import numpy as np
        import torch
    except ImportError:
        return False

    sys.path.insert(0, str(repo))
    try:
        from wan.configs import WAN_CONFIGS  # type: ignore
        from wan.modules.t5 import T5EncoderModel  # type: ignore
    except Exception as exc:
        print(f"wan import failed: {exc}", file=sys.stderr)
        return False

    cfg_wan = WAN_CONFIGS.get("t2v-1.3B") or next(iter(WAN_CONFIGS.values()))
    # Latent grid: Wan VAE stride t=4, h=w=8
    lt = (frames - 1) // 4 + 1
    lh = height // 8
    lw = width // 8
    z_channels = 16
    latent_shape = (z_channels, lt, lh, lw)

    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    if os.environ.get("WAN_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        device = torch.device("cpu")
    # Wan flash_attention requires CUDA; force SDPA on MPS/CPU for parity dumps.
    if device.type != "cuda" and not os.environ.get("WAN_FORCE_SDPA"):
        os.environ["WAN_FORCE_SDPA"] = "1"
        print("WAN_FORCE_SDPA=1 (non-CUDA DiT dump)", flush=True)

    emb_np = None
    if t5_from_c and noise_from is not None:
        emb_np = _load_t5_from_c(noise_from)
        if emb_np is not None:
            print(f"injected T5 from {noise_from} shape={emb_np.shape}", flush=True)

    if emb_np is None:
        t5_path = ckpt / "models_t5_umt5-xxl-enc-bf16.pth"
        tok_path = ckpt / "google" / "umt5-xxl"
        if not t5_path.is_file():
            print(f"missing T5 weights: {t5_path}", file=sys.stderr)
            return False

        print(f"loading T5 on {device} from {t5_path} …", flush=True)
        text_encoder = T5EncoderModel(
            text_len=getattr(cfg_wan, "text_len", 512),
            dtype=torch.float32,
            device=device,
            checkpoint_path=str(t5_path),
            tokenizer_path=str(tok_path) if tok_path.is_dir() else None,
        )
        with torch.no_grad():
            context = text_encoder([prompt], device)
        if isinstance(context, (list, tuple)):
            emb = context[0]
        else:
            emb = context[0] if context.ndim == 3 else context
        emb_np = emb.detach().float().cpu().numpy()
        del text_encoder
        if device.type == "mps":
            torch.mps.empty_cache()

    if noise_from is not None:
        noise = _load_noise_from_c(noise_from, latent_shape)
        print(f"injected noise from {noise_from} shape={noise.shape}", flush=True)
        # Prefer wan-c gen_t when present so DiT A/B shares the same t.
        try:
            c_meta = json.loads((noise_from / "meta.json").read_text())
            if c_meta.get("gen_t") is not None:
                os.environ.setdefault("WAN_PARITY_T", str(c_meta["gen_t"]))
        except Exception:
            pass
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        noise = torch.randn(
            z_channels, lt, lh, lw, dtype=torch.float32, generator=g
        ).numpy()

    meta = {
        "prompt": prompt,
        "ckpt_dir": str(ckpt),
        "steps": steps,
        "mode": "wan_t5_noise" + ("_dit" if dit else ""),
        "device": str(device),
        "t5_shape": list(emb_np.shape),
        "latent_shape": list(latent_shape),
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "cfg_scale": cfg,
        "shift": shift,
        "noise_from": str(noise_from) if noise_from else None,
        "t5_from_c": bool(t5_from_c and noise_from is not None),
        "configs": list(WAN_CONFIGS.keys())[:8],
    }

    dit_pred_np = None
    latent_s1_np = None
    if dit:
        try:
            from wan.modules.model import WanModel  # type: ignore
            from wan.utils.fm_solvers_unipc import (  # type: ignore
                FlowUniPCMultistepScheduler,
            )
        except Exception as exc:
            print(f"DiT import failed: {exc}", file=sys.stderr)
            return False

        print(f"loading WanModel from {ckpt} on {device} …", flush=True)
        model = WanModel.from_pretrained(str(ckpt))
        model.eval()
        model.to(device)
        # Match wan-c host path: prefer fp32 for parity.
        model.to(dtype=torch.float32)

        patch = getattr(cfg_wan, "patch_size", (1, 2, 2))
        sp_size = 1
        seq_len = (
            math.ceil(
                (lh * lw)
                / (patch[1] * patch[2])
                * lt
                / sp_size
            )
            * sp_size
        )

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=getattr(cfg_wan, "num_train_timesteps", 1000),
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(steps, device=device, shift=shift)
        timesteps = sample_scheduler.timesteps
        t0 = timesteps[0]
        model_t = t0
        override_t = os.environ.get("WAN_PARITY_T")
        if override_t:
            t_val = float(override_t)
            model_t = torch.tensor(t_val, device=device, dtype=torch.float32)
            print(f"WAN_PARITY_T override model t={t_val} (sched t={float(t0)})", flush=True)
        sigma0 = float(model_t.item()) / 1000.0
        meta["sigma0"] = sigma0
        meta["gen_t"] = float(model_t.item())
        meta["seq_len"] = int(seq_len)

        latent = torch.from_numpy(noise).to(device=device, dtype=torch.float32)
        context = [torch.from_numpy(emb_np).to(device=device, dtype=torch.float32)]
        context_null = None
        if cfg > 1.0001:
            # Match wan-c: empty-prompt encoding for uncond (not sample_neg_prompt).
            t5_path = ckpt / "models_t5_umt5-xxl-enc-bf16.pth"
            tok_path = ckpt / "google" / "umt5-xxl"
            text_encoder = T5EncoderModel(
                text_len=getattr(cfg_wan, "text_len", 512),
                dtype=torch.float32,
                device=device,
                checkpoint_path=str(t5_path),
                tokenizer_path=str(tok_path) if tok_path.is_dir() else None,
            )
            with torch.no_grad():
                null_ctx = text_encoder([""], device)
            if isinstance(null_ctx, (list, tuple)):
                null_emb = null_ctx[0]
            else:
                null_emb = null_ctx[0] if null_ctx.ndim == 3 else null_ctx
            context_null = [null_emb.detach().float()]
            del text_encoder

        with torch.no_grad():
            timestep = torch.stack([model_t.to(device)])
            pred_cond = model([latent], t=timestep, context=context, seq_len=seq_len)[
                0
            ]
            if context_null is not None:
                pred_uncond = model(
                    [latent], t=timestep, context=context_null, seq_len=seq_len
                )[0]
                noise_pred = pred_uncond + cfg * (pred_cond - pred_uncond)
            else:
                noise_pred = pred_cond

            dit_pred_np = noise_pred.detach().float().cpu().numpy()
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t0,
                latent.unsqueeze(0),
                return_dict=False,
                generator=None,
            )[0]
            latent_s1_np = temp_x0.squeeze(0).detach().float().cpu().numpy()

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        meta["dumped"] = [
            "t5_emb.npy",
            "noise_latent.npy",
            "dit_pred.npy",
            "latent_s1.npy",
        ]
        print(
            f"DiT step0 pred shape={dit_pred_np.shape} "
            f"sigma0={sigma0:.6f} t={meta['gen_t']:.2f} seq_len={seq_len}",
            flush=True,
        )

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "t5_emb.npy", emb_np.astype(np.float32))
    np.save(out / "noise_latent.npy", noise.astype(np.float32))
    if dit_pred_np is not None:
        np.save(out / "dit_pred.npy", dit_pred_np.astype(np.float32))
    if latent_s1_np is not None:
        np.save(out / "latent_s1.npy", latent_s1_np.astype(np.float32))
    print(
        f"wrote Wan dumps under {out} t5={emb_np.shape} noise={noise.shape}"
        + (f" dit={dit_pred_np.shape}" if dit_pred_np is not None else "")
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("dumps/parity_py"))
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dit", action="store_true", help="Also dump DiT step-0 pred")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument(
        "--noise-from",
        type=Path,
        default=None,
        help="Inject noise.f32 (+ optional t5) from wan-c WAN_DUMP_DIR",
    )
    ap.add_argument(
        "--t5-from-c",
        action="store_true",
        help="With --noise-from, also inject t5_emb.f32 (isolate DiT)",
    )
    args = ap.parse_args()

    ckpt = args.ckpt_dir or Path(
        os.environ.get(
            "WAN_CKPT",
            Path.home() / ".zerollama/third_party/wan/Wan2.1-T2V-1.3B",
        )
    )
    repo = args.repo or Path(
        os.environ.get(
            "WAN_REPO",
            Path.home() / ".zerollama/third_party/wan/Wan2.1",
        )
    )

    shape = (16, (args.frames - 1) // 4 + 1, args.height // 8, args.width // 8)
    if repo.is_dir() and try_wan_dump(
        args.out,
        args.prompt,
        args.steps,
        ckpt,
        repo,
        args.width,
        args.height,
        args.frames,
        args.seed,
        dit=args.dit,
        cfg=args.cfg,
        shift=args.shift,
        noise_from=args.noise_from,
        t5_from_c=args.t5_from_c,
    ):
        return
    write_stub(args.out, args.prompt, args.steps, ckpt, shape)


if __name__ == "__main__":
    main()
