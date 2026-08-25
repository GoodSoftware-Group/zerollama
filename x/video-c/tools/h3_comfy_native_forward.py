#!/usr/bin/env python3
"""Comfy MiniMaxH3 _forward on Comfy-built latents (not host H3_DUMP_L0).

128² T2VA, sigma=1, optional H3TE dump. Prints packed x_rms through 50 blocks
and final video/audio head RMS. Does not run 1344.

  ComfyUI/.venv/bin/python x/video-c/tools/h3_comfy_native_forward.py \\
    --text-cond /tmp/h3_te32_oracle.bin
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

# Kitchen INT8 uses aten::_int_mm, which PyTorch does not implement on MPS.
# Fallback keeps other ops on Metal; the i8 GEMM runs on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

COMFY = Path("/Users/user1/Sites/inference/ComfyUI")
PACK = COMFY / "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DIM = 5120


def rms(t: torch.Tensor) -> float:
    x = t.detach().float().cpu().numpy()
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def patch_row_div(v: torch.Tensor, patch_size) -> str:
    """Pairwise cos of first 8 video patch rows (host log_vid_div video_out)."""
    from comfy.ldm.minimax.model import patchify_video

    rows = patchify_video(v.float().cpu(), patch_size).float().numpy()
    n = min(8, rows.shape[0])
    acc = 0.0
    npair = 0
    for i in range(n):
        a = rows[i]
        na = float(np.sqrt(np.dot(a, a)))
        for j in range(i + 1, n):
            b = rows[j]
            nb = float(np.sqrt(np.dot(b, b)))
            if na > 1e-12 and nb > 1e-12:
                acc += float(np.dot(a, b) / (na * nb))
                npair += 1
    dlt = rows[0] - rows[1]
    r01 = float(np.sqrt(np.mean(dlt * dlt)))
    return f"vid-div video_out n={n} mean_cos={acc / npair if npair else 0:.4f} row0_vs_1_rms={r01:.4g}"


def spatial_stats(v: torch.Tensor) -> str:
    """Host h3_dit_log_latent_spatial: t=0 mean_C map, lag-1 ac1, mean per-ch std."""
    x = v.detach().float().cpu().numpy()
    if x.ndim == 5:
        x = x[0]
    _c, _t, h, w = x.shape
    plane = x[:, 0].mean(axis=0)
    m = float(plane.mean())
    d = plane - m
    var = float(np.mean(d * d))
    n1 = 0
    ac = 0.0
    for y in range(h):
        for xi in range(w - 1):
            ac += float(d[y, xi] * d[y, xi + 1])
            n1 += 1
    ac1 = (ac / n1) / var if n1 and var > 1e-20 else 0.0
    ch = 0.0
    for ci in range(x.shape[0]):
        p = x[ci, 0].reshape(-1)
        cm = float(p.mean())
        ch += float(np.sqrt(np.mean((p - cm) ** 2)))
    pch = ch / x.shape[0] if x.shape[0] else 0.0
    std = float(np.sqrt(var))
    return (
        f"latent spatial {w}x{h} (t=0 mean_C) std={std:.4g} ac1={ac1:.3f} "
        f"per-ch_std={pch:.4g}"
    )


def load_h3te(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    magic, nt, dim = struct.unpack_from("<4sII", blob, 0)
    if magic != b"H3TE" or dim != DIM:
        raise SystemExit(f"bad H3TE {path} magic={magic!r} dim={dim}")
    cond = np.frombuffer(blob, dtype=np.float32, offset=12, count=nt * dim)
    return cond.reshape(nt, dim).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-cond", default="/tmp/h3_te32_oracle.bin")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, default=50, help="0 = embed only")
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--video-latent", default="", help="f32 CTHW (host H3_VIDEO_LATENT)")
    ap.add_argument("--audio-latent", default="", help="f32 (2,C,T) (host H3_AUDIO_LATENT)")
    ap.add_argument("--host-vel", default="", help="f32 CTHW host vpred to cosine vs video_out")
    ap.add_argument(
        "--dtype",
        default="fp32",
        choices=("fp32", "bf16", "default"),
        help="default = UNETLoader weight_dtype (keep int8 QuantizedTensor)",
    )
    ap.add_argument(
        "--native-forward",
        action="store_true",
        help="Call MiniMaxH3.forward (oracle wrap: audio_scale) not _forward",
    )
    ap.add_argument(
        "--audio-scale",
        type=float,
        default=4.0,
        help="ModelSamplingAV shift12/audio3 = 4; 1 skips the wrap (old rematch)",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Video sigma for native forward (oracle last jump is 0.6316)",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=("auto", "mps", "cpu"),
        help="auto = mps if torch.backends.mps.is_available() else cpu",
    )
    ap.add_argument("--comfy", default=str(COMFY))
    args = ap.parse_args()

    comfy_root = Path(args.comfy).resolve()
    os.chdir(comfy_root)
    sys.path.insert(0, str(comfy_root))
    # Lab scripts do not enable Comfy args_parsing. Without --cpu, Comfy
    # tries CUDA on this Mac and dies; with --cpu it never uses MPS.
    want_mps = args.device == "mps" or (
        args.device == "auto" and torch.backends.mps.is_available()
    )
    from comfy.cli_args import args as comfy_args
    comfy_args.cpu = not want_mps

    import folder_paths  # noqa: F401
    import comfy.model_management as mm
    import comfy.sd
    from comfy.ldm.minimax.model import (
        PackedLayout,
        pack_audio,
        patchify_video,
        rope_rotation_table,
        time_shift_sigma,
        unpack_audio,
        unpatchify_video,
    )

    compute = None if args.dtype == "default" else (
        torch.bfloat16 if args.dtype == "bf16" else torch.float32
    )
    device = torch.device("mps" if want_mps else "cpu")
    print(
        f"device={device} mps_available={torch.backends.mps.is_available()} "
        f"comfy_cpu={bool(comfy_args.cpu)} "
        f"mps_fallback={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '')}",
        flush=True,
    )
    lw, lh = args.width // 16, args.height // 16
    latent_t, audio_t = 2, 8
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    video = torch.randn(1, 24, latent_t, lh, lw, generator=g, dtype=torch.float32)
    audio = torch.randn(1, 32, 2, audio_t, generator=g, dtype=torch.float32)
    if args.video_latent:
        raw = np.fromfile(args.video_latent, dtype=np.float32)
        want = 24 * latent_t * lh * lw
        if raw.size != want:
            raise SystemExit(f"video-latent {raw.size} want {want}")
        video = torch.from_numpy(raw.reshape(1, 24, latent_t, lh, lw).copy())
        print(f"loaded video-latent {args.video_latent} rms={rms(video):.6g}", flush=True)
    if args.audio_latent:
        raw = np.fromfile(args.audio_latent, dtype=np.float32)
        want = 2 * 32 * audio_t
        if raw.size != want:
            raise SystemExit(f"audio-latent {raw.size} want {want}")
        a2ct = raw.reshape(2, 32, audio_t)
        audio = torch.from_numpy(np.ascontiguousarray(a2ct.transpose(1, 0, 2)[None].copy()))
        print(f"loaded audio-latent {args.audio_latent} rms={rms(audio):.6g}", flush=True)

    te = Path(args.text_cond)
    if te.is_file():
        cond = load_h3te(te)
        context = torch.from_numpy(cond).unsqueeze(0)
        print(f"text-cond {te} nt={cond.shape[0]} rms={rms(context):.4g}", flush=True)
    else:
        nt = 12
        context = torch.zeros(1, nt, DIM)
        for i in range(nt * DIM):
            context.view(-1)[i] = 0.01 * np.sin(i * 0.02)
        print(f"dummy text nt={nt}", flush=True)

    print(f"loading {PACK} dtype={args.dtype}", flush=True)
    mo = {"load_device": device, "offload_device": device}
    if compute is not None:
        mo["dtype"] = compute
    patcher = comfy.sd.load_diffusion_model(str(PACK), model_options=mo)
    mm.load_models_gpu([patcher], force_full_load=True)
    dm = patcher.model.diffusion_model
    dm.eval()
    w0 = dm.blocks[0].mlp.fc1.weight
    print(
        f"fc1.weight type={type(w0).__name__} "
        f"layout={getattr(w0, '_layout_cls', None)} "
        f"quant={getattr(dm.blocks[0].mlp.fc1, 'quant_format', None)} "
        f"full_prec_mm={getattr(dm.blocks[0].mlp.fc1, '_full_precision_mm', None)}",
        flush=True,
    )

    sigma_v = float(args.sigma)

    with torch.no_grad():
        if args.native_forward:
            cdtype = None
            fk = getattr(dm.blocks[0].mlp.fc1, "factory_kwargs", None)
            if isinstance(fk, dict):
                cdtype = fk.get("dtype")
            if cdtype is None:
                cdtype = torch.bfloat16
            ctx = context.to(device=device, dtype=cdtype)
            timestep = torch.tensor([sigma_v * 1000.0], dtype=torch.float32, device=device)
            payload = {"audio_scale": float(args.audio_scale)}
            use_inner = args.audio_scale == 1.0
            fn = dm._forward if use_inner else dm.forward
            layout = PackedLayout(context.shape[1], latent_t, lh, lw, audio_t)
            va, vb = next((a, b) for a, b, k in layout.segments if k == "video")
            want_l = {23, 47, 48, 49}

            def _vid_cos(h, tag):
                rows = h[va:vb].float()
                n = min(8, rows.shape[0])
                acc = 0.0
                npair = 0
                for i in range(n):
                    a = rows[i]
                    na = float(torch.linalg.vector_norm(a))
                    for j in range(i + 1, n):
                        b = rows[j]
                        nb = float(torch.linalg.vector_norm(b))
                        if na > 1e-12 and nb > 1e-12:
                            acc += float(torch.dot(a, b) / (na * nb))
                            npair += 1
                print(
                    f"comfy vid-div {tag} n={n} mean_cos={acc / npair if npair else 0:.4f} "
                    f"x_rms={rms(h):.6g}",
                    flush=True,
                )

            orig_fwd = []
            for li, blk in enumerate(dm.blocks):
                orig_fwd.append(blk.forward)

                def make(i, old):
                    def wrapped(*a, **k):
                        y = old(*a, **k)
                        if i in want_l:
                            _vid_cos(y, f"after_L{i}")
                        return y

                    return wrapped

                blk.forward = make(li, blk.forward)

            print(
                f"native {fn.__name__} {args.width}x{args.height} ctx_dtype={ctx.dtype} "
                f"sigma={sigma_v} layers={len(dm.blocks)} audio_scale={payload['audio_scale']} "
                f"video_rows={va}:{vb}",
                flush=True,
            )
            out = fn(
                [video.to(device), audio.to(device)],
                timestep,
                ctx,
                transformer_options={},
                minimax_payload=payload,
            )
            vneg, aneg = out[0], out[1]
            print(
                f"native video_out_neg_rms={rms(vneg):.6g} audio_out_neg_rms={rms(aneg):.6g} "
                f"vshape={tuple(vneg.shape)}",
                flush=True,
            )
            print(f"native {spatial_stats(vneg)}", flush=True)
            print(f"native {patch_row_div(vneg, dm.patch_size)}", flush=True)
            if args.dump_dir:
                dump = Path(args.dump_dir)
                dump.mkdir(parents=True, exist_ok=True)
                video[0].contiguous().cpu().numpy().astype(np.float32).tofile(
                    dump / "video_cthw.bin")
                audio[0].permute(1, 0, 2).contiguous().cpu().numpy().astype(
                    np.float32).tofile(dump / "audio_2ct.bin")
                vneg[0].float().cpu().numpy().astype(np.float32).tofile(
                    dump / "video_out.bin")
                print(f"wrote {dump}", flush=True)
            if args.host_vel:
                hv = np.fromfile(args.host_vel, dtype=np.float32)
                cv = vneg[0].float().cpu().numpy().reshape(-1)
                if hv.size != cv.size:
                    print(f"host-vel size {hv.size} vs comfy {cv.size}", flush=True)
                else:
                    a = hv.astype(np.float64)
                    b = cv.astype(np.float64)
                    na = float(np.sqrt(np.dot(a, a)))
                    nb = float(np.sqrt(np.dot(b, b)))
                    cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
                    d = a - b
                    dn = a + b
                    print(
                        f"host vs comfy video_out cosine={cos:.6f} "
                        f"rel_rms={float(np.sqrt(np.mean(d * d))) / (nb / np.sqrt(b.size) + 1e-12):.4g} "
                        f"cosine_neg={float(np.dot(a, -b) / (na * nb)) if na > 0 and nb > 0 else 0:.6f} "
                        f"max_abs={float(np.max(np.abs(d))):.4g}",
                        flush=True,
                    )
            return 0

        if compute is None:
            raise SystemExit("manual replay needs --dtype fp32|bf16 (or use --native-forward)")

        layout = PackedLayout(context.shape[1], latent_t, lh, lw, audio_t)
        shift_v, shift_a = 12.0, 3.0
        t_v = 1.0 - sigma_v
        t_a = float(1.0 - time_shift_sigma(torch.tensor(sigma_v), shift_v, shift_a))
        unique_t = sorted({t_v, t_a})
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2}
        segs = []
        for a, b, kind in layout.segments:
            segs.append((a, b, t_row[{"text": t_v, "video": t_v, "audio": t_a}[kind]] * 3
                         + seg_tag[kind]))

        video_rows = patchify_video(video, dm.patch_size)
        audio_rows = pack_audio(audio)
        video_embed = dm.video_patch_proj(video_rows).to(compute)
        audio_embed = dm.audio_patch_proj(audio_rows).to(compute)
        text_states = context[0].to(device=device, dtype=compute)
        if text_states.shape[-1] != dm.hidden_size:
            text_states = dm.token_refiner(
                dm.condition_proj(text_states), transformer_options={})

        h = torch.empty(layout.seq_len, dm.hidden_size, dtype=compute, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind == "video":
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n
        print(
            f"embed seq={layout.seq_len} nv={video_rows.shape[0]} "
            f"packed_rms={rms(h):.6g} video_embed_rms={rms(video_embed):.6g}",
            flush=True,
        )
        dump = Path(args.dump_dir) if args.dump_dir else None
        if dump:
            dump.mkdir(parents=True, exist_ok=True)
            h.float().cpu().numpy().tofile(dump / "packed.bin")
            layout.position_ids.float().cpu().numpy().tofile(dump / "pos.bin")
            video[0].contiguous().cpu().numpy().tofile(dump / "video_cthw.bin")
            # host pack_audio wants (2, C, T)
            audio[0].permute(1, 0, 2).contiguous().cpu().numpy().tofile(
                dump / "audio_2ct.bin")
            (dump / "meta.txt").write_text(
                f"seq={layout.seq_len} H={dm.hidden_size} nv={video_rows.shape[0]} "
                f"w={args.width} h={args.height} nt={context.shape[1]}\n")
            print(f"wrote {dump}", flush=True)

        if args.layers < 1:
            return 0

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        table = dm.adaln_t_table.to(device=device, dtype=torch.float32)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        inv = dm.rope.inv_freq.to(device=device, dtype=torch.float32)
        per = layout.position_ids.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv.view(1, 1, -1)
        t_f, h_f, w_f = per.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        rope_freqs = rope_rotation_table(torch.cat((half, half), dim=-1), compute)

        want = {0, 23, 35, 45, 46, 47, 48, 49}
        n_run = min(args.layers, len(dm.blocks))
        for li, block in enumerate(dm.blocks[:n_run]):
            h = block(h, t_emb, segs, rope_freqs)
            if li in want:
                print(f"comfy-built L{li} x_rms={rms(h):.6g}", flush=True)

        video_seg = next((a, b, t_row[t_v]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[t_a]) for a, b, k in layout.segments if k == "audio")
        v, a = dm.final_layer(h, t_emb, video_seg, audio_seg)
        print(f"replay video_head_rms={rms(v):.6g} audio_head_rms={rms(a):.6g}", flush=True)
        vpix = unpatchify_video(v, latent_t, lh // 2, lw // 2, dm.latents_dim, dm.patch_size)
        print(f"replay unpatch rms={rms(vpix):.6g} unpack_a={rms(unpack_audio(a)):.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
