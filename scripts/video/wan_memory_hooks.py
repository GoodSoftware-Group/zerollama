"""Host-RAM hooks for Wan T2V/TI2V on tight containers (~16 GB CT).

T5-XXL stays loaded for the whole upstream job (~11 GB on CPU with t5_cpu).
We release it after cond+uncond prompts are encoded. Optional CPU VAE decode
duplicates weights on host — default off when targeting low CT RAM.

Wan2.1 exposes ``wan.modules.vae.WanVAE``; Wan2.2 renamed that to
``Wan2_1_VAE`` / ``Wan2_2_VAE`` (TI2V uses 2.2). Hooks resolve both so a single
wrapper works across profiles.
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


def mmgp_enabled() -> bool:
    """Layer/budget VRAM offload via mmgp (WanGP). Default off unless job/env sets WAN_MMGP=1."""
    return os.environ.get("WAN_MMGP", "0").lower() in ("1", "true", "yes")


def mmgp_profile_no() -> int:
    raw = os.environ.get("WAN_MMGP_PROFILE", "5").strip() or "5"
    try:
        return int(float(raw))
    except ValueError:
        return 5


def mmgp_quantize_transformer() -> bool:
    return os.environ.get("WAN_MMGP_QUANTIZE", "0").lower() in ("1", "true", "yes")


def import_mmgp_early() -> bool:
    """Import mmgp before safetensors-heavy loads (mmgp README). Returns True if imported."""
    if not mmgp_enabled():
        return False
    try:
        import mmgp  # noqa: F401
        from mmgp import offload  # noqa: F401

        print("WAN: mmgp imported early (safetensors redirect)", file=sys.stderr, flush=True)
        return True
    except Exception as err:
        print(f"WAN: mmgp import failed ({err}) — continuing without", file=sys.stderr, flush=True)
        return False


def _neutralize_full_dit_device_moves(model) -> None:
    """No-op whole-module .to(cuda|cpu) / .cpu() so stock Wan offload_model cannot reload full DiT."""
    if model is None or getattr(model, "_zerollama_mmgp_neutralized", False):
        return
    import torch

    _orig_to = model.to

    def to(*args, **kwargs):  # type: ignore[no-untyped-def]
        target = None
        if args:
            target = args[0]
        elif "device" in kwargs:
            target = kwargs["device"]
        # Allow dtype-only casts (convert_model_dtype path).
        if target is None or isinstance(target, torch.dtype):
            return _orig_to(*args, **kwargs)
        try:
            dev = target if isinstance(target, torch.device) else torch.device(target)
            if dev.type in ("cuda", "cpu", "mps"):
                print(
                    f"WAN: mmgp — skipped full DiT .to({dev})",
                    file=sys.stderr,
                    flush=True,
                )
                return model
        except Exception:
            pass
        return _orig_to(*args, **kwargs)

    model.to = to  # type: ignore[method-assign]
    model._zerollama_mmgp_neutralized = True


def attach_mmgp_profile(pipe) -> None:
    """Wrap WanTI2V/WanT2V submodules with mmgp offload.profile (Wan2GP pipe layout)."""
    if not mmgp_enabled() or getattr(pipe, "_zerollama_mmgp", False):
        return
    try:
        from mmgp import offload
    except Exception as err:
        print(f"WAN: mmgp not available ({err})", file=sys.stderr, flush=True)
        return

    dit = getattr(pipe, "model", None)
    vae_wrap = getattr(pipe, "vae", None)
    te_wrap = getattr(pipe, "text_encoder", None)
    vae_core = getattr(vae_wrap, "model", None) if vae_wrap is not None else None
    te_core = getattr(te_wrap, "model", None) if te_wrap is not None else None
    if dit is None:
        print("WAN: mmgp skip — no DiT on pipe", file=sys.stderr, flush=True)
        return

    pipe_dict = {"transformer": dit}
    # Keep VAE out of mmgp when WAN_VAE_CPU=1 — our encode/decode hooks own CPU residency;
    # putting VAE in the pipe fights scale/latent device placement (cuda vs cpu).
    if vae_core is not None and not vae_cpu_enabled():
        pipe_dict["vae"] = vae_core
    if te_core is not None:
        pipe_dict["text_encoder"] = te_core

    profile = mmgp_profile_no()
    quantize = mmgp_quantize_transformer()
    try:
        offloadobj = offload.profile(
            pipe_dict,
            profile_no=profile,
            quantizeTransformer=quantize,
            verboseLevel=1,
        )
        pipe._zerollama_mmgp = True
        pipe._zerollama_mmgp_offload = offloadobj
        _neutralize_full_dit_device_moves(dit)
        alloc = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 2**30
        except Exception:
            pass
        print(
            f"WAN: mmgp profile={profile} quantizeTransformer={quantize} "
            f"cuda_alloc={alloc:.2f}GiB keys={list(pipe_dict)}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as err:
        print(f"WAN: mmgp.profile failed: {err}", file=sys.stderr, flush=True)


def patch_mmgp_fp32_time_mod() -> None:
    """Stock Wan asserts time-modulation tensors are fp32; mmgp + convert_model_dtype can yield bf16.

    WanGP forks comment out those asserts and adapt modulation dtypes. We cast ``e`` to
    float32 at the block/head boundary instead of editing upstream sources.
    """
    if not mmgp_enabled():
        return
    import torch

    try:
        from wan.modules.model import Head, WanAttentionBlock
    except Exception as err:
        print(f"WAN: cannot patch fp32 time-mod for mmgp ({err})", file=sys.stderr, flush=True)
        return

    def _as_fp32(e):  # type: ignore[no-untyped-def]
        if torch.is_tensor(e) and e.dtype != torch.float32:
            return e.float()
        return e

    _head = Head.forward

    def head_forward(self, x, e):  # type: ignore[no-untyped-def]
        return _head(self, x, _as_fp32(e))

    Head.forward = head_forward  # type: ignore[method-assign]

    _block = WanAttentionBlock.forward

    def block_forward(self, x, e, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _block(self, x, _as_fp32(e), *args, **kwargs)

    WanAttentionBlock.forward = block_forward  # type: ignore[method-assign]
    print("WAN: mmgp fp32 time-modulation cast armed", file=sys.stderr, flush=True)


def patch_mmgp_profile() -> None:
    """After pipeline __init__, attach mmgp (must run after VAE-CPU coerce wraps)."""
    if not mmgp_enabled():
        return

    for mod_name, cls_name in (
        ("wan.textimage2video", "WanTI2V"),
        ("wan.text2video", "WanT2V"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception as err:
            print(f"WAN: cannot patch {cls_name} for mmgp ({err})", file=sys.stderr, flush=True)
            continue

        _orig_init = cls.__init__

        def __init__(self, *args, _orig=_orig_init, _cls=cls_name, **kwargs):  # type: ignore[no-untyped-def]
            _orig(self, *args, **kwargs)
            try:
                attach_mmgp_profile(self)
            except Exception as err:
                print(f"WAN: mmgp attach on {_cls} failed: {err}", file=sys.stderr, flush=True)

        cls.__init__ = __init__  # type: ignore[method-assign]
        print(f"WAN: mmgp hook armed for {cls_name}", file=sys.stderr, flush=True)


def _pipeline_classes() -> list[type]:
    """Wan generate entrypoints to wrap (T2V and TI2V when present)."""
    out: list[type] = []
    try:
        from wan.text2video import WanT2V

        out.append(WanT2V)
    except Exception as err:
        print(f"WAN: WanT2V unavailable ({err})", file=sys.stderr, flush=True)
    try:
        from wan.textimage2video import WanTI2V

        out.append(WanTI2V)
    except Exception as err:
        print(f"WAN: WanTI2V unavailable ({err})", file=sys.stderr, flush=True)
    return out


def _vae_wrapper_classes() -> list[type]:
    """Public VAE wrapper classes (decode on these)."""
    out: list[type] = []
    for mod_name, attr in (
        ("wan.modules.vae", "WanVAE"),
        ("wan.modules.vae2_1", "Wan2_1_VAE"),
        ("wan.modules.vae2_2", "Wan2_2_VAE"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[attr])
            cls = getattr(mod, attr)
            out.append(cls)
        except Exception:
            continue
    return out


def _vae_core_classes() -> list[type]:
    """Internal WanVAE_ cores used for per-frame decode progress.

    Skip ``vae2_2`` — its temporal residual layout breaks the Wan2.1-style
    per-frame feat_cache loop (RuntimeError on shortcut add). Wrapper-level
    PROGRESS in ``patch_vae_cpu_decode`` covers TI2V decode instead.
    """
    out: list[type] = []
    for mod_name in ("wan.modules.vae", "wan.modules.vae2_1"):
        try:
            mod = __import__(mod_name, fromlist=["WanVAE_"])
            out.append(getattr(mod, "WanVAE_"))
        except Exception:
            continue
    return out


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


def cast_dit_float32_if_needed(pipe) -> None:
    """After T5 is gone, materialize DiT in fp32 for Darwin CPU/MPS."""
    import torch

    if torch.cuda.is_available():
        return
    if not getattr(pipe, "_zerollama_need_fp32", False):
        return
    if not hasattr(pipe, "model") or pipe.model is None:
        return
    pipe.model = pipe.model.to(dtype=torch.float32)
    pipe.param_dtype = torch.float32
    pipe._zerollama_need_fp32 = False
    print("WAN: cast DiT to float32 (Apple Silicon bf16 path)", file=sys.stderr, flush=True)


def patch_unload_t5_after_encode() -> None:
    """Unload T5 after encode (16g) and cast DiT to fp32 on Darwin once T5 is gone."""
    import torch

    do_unload = unload_t5_enabled()
    for pipe_cls in _pipeline_classes():
        _orig_gen = pipe_cls.generate

        def generate(self, *args, _orig=_orig_gen, **kwargs):  # type: ignore[no-untyped-def]
            te = getattr(self, "text_encoder", None)
            if te is None:
                cast_dit_float32_if_needed(self)
                return _orig(self, *args, **kwargs)

            _orig_call = te.__call__
            encode_passes = [0]

            def wrapped_call(texts, device):
                out = _orig_call(texts, device)
                if not torch.cuda.is_available():
                    target = torch.float32 if getattr(self, "_zerollama_need_fp32", False) else getattr(
                        self, "param_dtype", torch.float32
                    )
                    out = [t.to(dtype=target) for t in out]
                encode_passes[0] += 1
                # Cond + uncond: release after the second encode.
                if do_unload and encode_passes[0] >= 2:
                    te.__call__ = _orig_call
                    release_text_encoder(self)
                    cast_dit_float32_if_needed(self)
                return out

            te.__call__ = wrapped_call  # type: ignore[method-assign]
            try:
                return _orig(self, *args, **kwargs)
            finally:
                if getattr(self, "text_encoder", None) is not None:
                    te.__call__ = _orig_call
                cast_dit_float32_if_needed(self)

        pipe_cls.generate = generate  # type: ignore[method-assign]


def patch_vae_decode_progress() -> None:
    """Emit PROGRESS per latent frame during VAE decode (91–93% band)."""
    import threading
    import time

    import torch

    cores = _vae_core_classes()
    if not cores:
        print("WAN: no WanVAE_ core to patch for decode progress", file=sys.stderr, flush=True)
        return

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

    for core in cores:
        core.decode = decode  # type: ignore[method-assign]


def patch_vae_cpu_decode() -> None:
    if not vae_cpu_enabled():
        return
    import traceback

    import torch

    wrappers = _vae_wrapper_classes()
    if not wrappers:
        print("WAN: no VAE wrapper to patch for CPU decode", file=sys.stderr, flush=True)
        return

    for cls in wrappers:
        _orig_decode = cls.decode

        def decode(self, zs, _orig=_orig_decode):  # type: ignore[no-untyped-def]
            # Always decode on CPU when WAN_VAE_CPU=1 (MPS VAE is fragile / VRAM-heavy).
            try:
                gc.collect()
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                self.model.to("cpu")
                # Wan2.1 wrappers expose mean/std; Wan2.2 keeps only scale=[mean, 1/std].
                if hasattr(self, "mean") and hasattr(self, "std"):
                    self.mean = self.mean.cpu().float()
                    self.std = self.std.cpu().float()
                    self.scale = [self.mean, 1.0 / self.std]
                elif hasattr(self, "scale") and isinstance(self.scale, list):
                    self.scale = [
                        t.cpu().float() if torch.is_tensor(t) else t for t in self.scale
                    ]
                if hasattr(self, "device"):
                    self.device = "cpu"
                zs_cpu = [z.detach().to("cpu").float().contiguous() for z in zs]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print("WAN: VAE decode on CPU", file=sys.stderr, flush=True)
                print("PROGRESS:91.0:vae decode starting", flush=True)
                out = _orig(self, zs_cpu)
                print("PROGRESS:93.5:vae decode complete", flush=True)
                return out
            except Exception:
                traceback.print_exc()
                raise

        cls.decode = decode  # type: ignore[method-assign]


def patch_free_dit_before_vae() -> None:
    """Drop DiT weights before VAE decode on Darwin to cut peak RAM."""
    import torch

    if torch.cuda.is_available():
        return
    for pipe_cls in _pipeline_classes():
        _orig = pipe_cls.generate

        def generate(self, *args, _orig=_orig, **kwargs):  # type: ignore[no-untyped-def]
            vae = getattr(self, "vae", None)
            if vae is None:
                return _orig(self, *args, **kwargs)
            _decode = vae.decode

            def decode_free(zs):  # type: ignore[no-untyped-def]
                model = getattr(self, "model", None)
                if model is not None:
                    try:
                        model.cpu()
                    except Exception:
                        pass
                    self.model = None
                    del model
                    gc.collect()
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    print("WAN: freed DiT before VAE decode", file=sys.stderr, flush=True)
                return _decode(zs)

            vae.decode = decode_free  # type: ignore[method-assign]
            try:
                return _orig(self, *args, **kwargs)
            finally:
                vae.decode = _decode  # type: ignore[method-assign]

        pipe_cls.generate = generate  # type: ignore[method-assign]


def patch_load_progress() -> None:
    """Emit PROGRESS during Wan weight load (5–28% band)."""
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel

    wrappers = _vae_wrapper_classes()

    _orig_t5 = T5EncoderModel.__init__

    def t5_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        print("PROGRESS:12.0:loading T5 encoder", flush=True)
        return _orig_t5(self, *args, **kwargs)

    T5EncoderModel.__init__ = t5_init  # type: ignore[method-assign]

    for WanVAE in wrappers:
        _orig_vae = WanVAE.__init__

        def vae_init(self, *args, _orig=_orig_vae, **kwargs):  # type: ignore[no-untyped-def]
            print("PROGRESS:16.0:loading VAE", flush=True)
            return _orig(self, *args, **kwargs)

        WanVAE.__init__ = vae_init  # type: ignore[method-assign]

    _orig_from_pretrained = WanModel.from_pretrained

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        print("PROGRESS:20.0:loading diffusion model", flush=True)
        return _orig_from_pretrained(*args, **kwargs)

    WanModel.from_pretrained = from_pretrained  # type: ignore[method-assign]


def patch_vae_init_on_cpu() -> None:
    """Force Wan2.x VAE wrappers onto CPU at construct time when WAN_VAE_CPU=1.

    WHY: TI2V loads VAE on CUDA then ``model.to(cuda)`` for diffusion. On 16 GB
    that peaks ~10 GiB+ before the DiT move and OOMs even when idle nvidia-smi
    only shows the serve process (~260 MiB). Decode still runs on CPU via
    ``patch_vae_cpu_decode``; encode-on-CPU is acceptable for 16g drafts.
    """
    if not vae_cpu_enabled():
        return
    import torch

    for cls in _vae_wrapper_classes():
        _orig = cls.__init__

        def __init__(self, *args, _orig=_orig, **kwargs):  # type: ignore[no-untyped-def]
            if "device" in kwargs:
                kwargs["device"] = "cpu"
            # Positional device is uncommon; still coerce after if needed.
            _orig(self, *args, **kwargs)
            try:
                if hasattr(self, "model") and self.model is not None:
                    self.model.to("cpu")
                if hasattr(self, "device"):
                    self.device = "cpu"
                if hasattr(self, "mean"):
                    self.mean = self.mean.cpu()
                if hasattr(self, "std"):
                    self.std = self.std.cpu()
                if hasattr(self, "scale") and isinstance(self.scale, list):
                    self.scale = [
                        t.cpu() if torch.is_tensor(t) else t for t in self.scale
                    ]
                print("WAN: VAE constructed on CPU (16g peak)", file=sys.stderr, flush=True)
            except Exception as err:
                print(f"WAN: VAE CPU coerce failed: {err}", file=sys.stderr, flush=True)

        cls.__init__ = __init__  # type: ignore[method-assign]


def patch_ti2v_keep_vae_off_gpu() -> None:
    """Keep WanTI2V VAE on CPU through encode so DiT ``.to(cuda)`` fits on 16 GB.

    Idle nvidia-smi often shows only the serve PID (~260 MiB). The OOM is still
    real: TI2V puts the image on CUDA then ``vae.encode`` pulls the VAE onto the
    GPU; together with the ~10 GiB DiT move that exceeds 15 GiB.
    """
    if not vae_cpu_enabled():
        return
    import torch

    try:
        from wan.textimage2video import WanTI2V
    except Exception as err:
        print(f"WAN: cannot patch WanTI2V ({err})", file=sys.stderr, flush=True)
        return

    _orig_init = WanTI2V.__init__

    def _force_vae_cpu(pipe) -> None:
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        if hasattr(vae, "model") and vae.model is not None:
            vae.model.to("cpu")
        for attr in ("mean", "std"):
            t = getattr(vae, attr, None)
            if torch.is_tensor(t):
                setattr(vae, attr, t.cpu())
        if hasattr(vae, "scale") and isinstance(vae.scale, list):
            vae.scale = [t.cpu() if torch.is_tensor(t) else t for t in vae.scale]
        if hasattr(vae, "device"):
            vae.device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _orig_init(self, *args, **kwargs)
        try:
            _force_vae_cpu(self)
            vae = self.vae
            if vae is not None and not getattr(vae, "_zerollama_cpu_encode", False):
                _orig_encode = vae.encode

                def encode_cpu(videos):  # type: ignore[no-untyped-def]
                    _force_vae_cpu(self)
                    vids = []
                    for v in videos:
                        vids.append(v.detach().to("cpu") if torch.is_tensor(v) else v)
                    out = _orig_encode(vids)
                    # Move latents (small) to CUDA for the diffusion mix; keep VAE weights on CPU.
                    dev = getattr(self, "device", torch.device("cuda"))
                    moved = []
                    for o in out or []:
                        moved.append(o.to(dev) if torch.is_tensor(o) else o)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"WAN: VAE encode on CPU; cuda_alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB",
                        file=sys.stderr,
                        flush=True,
                    )
                    return moved

                vae.encode = encode_cpu  # type: ignore[method-assign]
                vae._zerollama_cpu_encode = True
            print(
                f"WAN: WanTI2V VAE on CPU; cuda_alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB",
                file=sys.stderr,
                flush=True,
            )
        except Exception as err:
            print(f"WAN: WanTI2V VAE→CPU failed: {err}", file=sys.stderr, flush=True)

    WanTI2V.__init__ = __init__  # type: ignore[method-assign]

    _orig_gen = WanTI2V.generate

    def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _force_vae_cpu(self)
        return _orig_gen(self, *args, **kwargs)

    WanTI2V.generate = generate  # type: ignore[method-assign]


def apply_memory_hooks() -> None:
    # Order matters: VAE device → TI2V VAE CPU → mmgp (after submodules exist) →
    # unload/cast → free-DiT-before-VAE → load progress → VAE decode.
    patch_vae_init_on_cpu()
    patch_ti2v_keep_vae_off_gpu()
    patch_mmgp_fp32_time_mod()
    patch_mmgp_profile()
    patch_unload_t5_after_encode()
    patch_free_dit_before_vae()
    patch_load_progress()
    # Heartbeat VAE progress nests poorly with CPU decode on Darwin.
    if sys.platform != "darwin":
        patch_vae_decode_progress()
    if vae_cpu_enabled():
        patch_vae_cpu_decode()
        print("WAN: VAE CPU decode enabled", file=sys.stderr, flush=True)
