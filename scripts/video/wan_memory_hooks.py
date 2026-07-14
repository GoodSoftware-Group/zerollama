"""Host-RAM hooks for Wan T2V on tight containers (~16 GB CT).

T5-XXL stays loaded for the whole upstream job (~11 GB on CPU with t5_cpu).
We release it after cond+uncond prompts are encoded. Optional CPU VAE decode
duplicates weights on host — default off when targeting low CT RAM.
"""
from __future__ import annotations

import gc
import os
import sys


def unload_t5_enabled() -> bool:
    return os.environ.get("WAN_UNLOAD_T5", "1").lower() not in ("0", "false", "no")


def vae_cpu_enabled() -> bool:
    default = "1"
    raw = os.environ.get("WAN_VAE_CPU", default)
    return raw.lower() in ("1", "true", "yes")


def release_text_encoder(pipe) -> None:
    import torch

    te = getattr(pipe, "text_encoder", None)
    if te is None:
        return
    if hasattr(te, "model"):
        del te.model
    if hasattr(te, "tokenizer"):
        del te.tokenizer
    pipe.text_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("WAN: released T5 encoder (~11G host RAM)", file=sys.stderr, flush=True)


def patch_unload_t5_after_encode() -> None:
    if not unload_t5_enabled():
        return
    from wan.text2video import WanT2V

    _orig_gen = WanT2V.generate

    def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        te = getattr(self, "text_encoder", None)
        if te is None:
            return _orig_gen(self, *args, **kwargs)

        _orig_call = te.__call__
        encode_passes = [0]

        def wrapped_call(texts, device):
            out = _orig_call(texts, device)
            encode_passes[0] += 1
            if encode_passes[0] >= 2:
                release_text_encoder(self)
            return out

        te.__call__ = wrapped_call
        try:
            return _orig_gen(self, *args, **kwargs)
        finally:
            if getattr(self, "text_encoder", None) is not None:
                te.__call__ = _orig_call

    WanT2V.generate = generate  # type: ignore[method-assign]


def patch_vae_decode_progress() -> None:
    """Emit PROGRESS per latent frame during VAE decode (91–93% band)."""
    import threading
    import time

    import torch
    from wan.modules.vae import WanVAE_

    def _decode_frame_with_heartbeat(
        frame_idx: int,
        frame_total: int,
        decode_fn,
    ):
        base = 91.0 + (frame_idx / frame_total) * 2.0
        end = 91.0 + ((frame_idx + 1) / frame_total) * 2.0
        span = end - base
        stop = threading.Event()
        est_sec = float(os.environ.get("WAN_VAE_FRAME_EST_SEC", "90"))
        print(
            f"PROGRESS:{base:.1f}:decoding frame {frame_idx + 1}/{frame_total}",
            flush=True,
        )

        def heartbeat() -> None:
            start = time.monotonic()
            while not stop.wait(10):
                elapsed = time.monotonic() - start
                frac = min(0.95, elapsed / est_sec)
                pct = base + span * frac
                print(
                    f"PROGRESS:{pct:.1f}:decoding frame {frame_idx + 1}/{frame_total}",
                    flush=True,
                )

        t = threading.Thread(target=heartbeat, daemon=True)
        t.start()
        try:
            return decode_fn()
        finally:
            stop.set()
            print(
                f"PROGRESS:{end:.1f}:decoding frame {frame_idx + 1}/{frame_total}",
                flush=True,
            )

    def decode(self, z, scale):  # type: ignore[no-untyped-def]
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = int(z.shape[2])
        x = self.conv2(z)
        out = None
        for i in range(iter_):
            self._conv_idx = [0]

            def run_frame():
                nonlocal out
                if i == 0:
                    out = self.decoder(
                        x[:, :, i:i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx)
                    return out
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
                return out

            if iter_ > 1:
                out = _decode_frame_with_heartbeat(i, iter_, run_frame)
            else:
                out = run_frame()
                print("PROGRESS:93.0:decoding frame 1/1", flush=True)
        self.clear_cache()
        return out

    WanVAE_.decode = decode  # type: ignore[method-assign]


def patch_vae_cpu_decode() -> None:
    if not vae_cpu_enabled():
        return
    import torch
    from wan.modules.vae import WanVAE

    def decode(self, zs):  # type: ignore[no-untyped-def]
        dev = self.device
        if isinstance(dev, str):
            dev = torch.device(dev)
        if dev.type != "cuda":
            with torch.no_grad():
                out = []
                for u in zs:
                    dec = self.model.decode(u.unsqueeze(0), self.scale)
                    out.append(dec.float().clamp_(-1, 1).squeeze(0))
                return out

        self.model.to("cpu")
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        self.scale = [self.mean, 1.0 / self.std]
        zs_cpu = [z.cpu().float() for z in zs]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("WAN: VAE decode on CPU", file=sys.stderr, flush=True)
        # Avoid WanVAE.decode's cuda amp.autocast — it can pull CPU decode onto GPU.
        with torch.no_grad():
            out = []
            for u in zs_cpu:
                dec = self.model.decode(u.unsqueeze(0), self.scale)
                out.append(dec.float().clamp_(-1, 1).squeeze(0))
            return out

    WanVAE.decode = decode  # type: ignore[method-assign]


def patch_load_progress() -> None:
    """Emit PROGRESS during WanT2V weight load (5–28% band)."""
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE

    _orig_t5 = T5EncoderModel.__init__

    def t5_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        print("PROGRESS:12.0:loading T5 encoder", flush=True)
        return _orig_t5(self, *args, **kwargs)

    _orig_vae = WanVAE.__init__

    def vae_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        print("PROGRESS:16.0:loading VAE", flush=True)
        return _orig_vae(self, *args, **kwargs)

    _orig_from_pretrained = WanModel.from_pretrained

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        print("PROGRESS:20.0:loading diffusion model", flush=True)
        return _orig_from_pretrained(*args, **kwargs)

    T5EncoderModel.__init__ = t5_init  # type: ignore[method-assign]
    WanVAE.__init__ = vae_init  # type: ignore[method-assign]
    WanModel.from_pretrained = from_pretrained  # type: ignore[method-assign]


def apply_memory_hooks() -> None:
    patch_unload_t5_after_encode()
    patch_load_progress()
    patch_vae_decode_progress()
    if vae_cpu_enabled():
        patch_vae_cpu_decode()
        print("WAN: VAE CPU decode enabled", file=sys.stderr, flush=True)
