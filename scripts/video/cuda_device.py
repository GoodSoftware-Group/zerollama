"""Dual-GPU placement: best-fit first; migrate movable residents only under contention.

Policy (ZEROLLAMA_CUDA_POLICY, default best_fit):
  1. Prefer a GPU where free_mib >= need (best-fit = smallest free that still fits).
     That packs compute and keeps the largest open chunk for future DiT jobs.
  2. IFF no GPU fits, reclaim *solvable* contention (irodori TTS we own) by moving
     it onto the other card, then place the job on the freed GPU.
  3. Desktop/compositor PIDs are not movable — never pretend we can.

Overrides: CUDA_VISIBLE_DEVICES, or LTX_CUDA_DEVICE / H3_CUDA_DEVICE / WAN_CUDA_DEVICE.
Need MiB: VIDEO_VRAM_NEED_MIB / LTX_VRAM_NEED_MIB / … or infer from model env.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# Sticky TTS sidecar we can relocate (same uid as video wrappers on astra).
_MOVABLE_CMDLINE_MARKERS = ("irodori_openai_tts", "irodori-tts")
# Never treat as reclaimable even if they use a lot of VRAM.
_IMMOVABLE_MARKERS = (
    "Xwayland",
    "Xorg",
    "kwin",
    "plasmashell",
    "steam",
    "steamwebhelper",
    "gnome-shell",
    "zerollama",
    "llama-server",
    "ollama",
)

_IRODORI_CUDA_ENV = _env(
    "ZEROLLAMA_IRODORI_CUDA_ENV",
    "/mnt/ollama_img/speech/irodori-cuda.env",
)


@dataclass
class GpuSnap:
    index: str
    free_mib: int
    used_mib: int
    movable: list[tuple[int, int, str]]  # (pid, used_mib, name)


def gpu_free_mib() -> list[tuple[str, int]]:
    """Return [(index, free_mib), ...] from nvidia-smi; empty on failure."""
    return [(g.index, g.free_mib) for g in snapshot_gpus()]


def snapshot_gpus() -> list[GpuSnap]:
    """Per-GPU free/used plus compute PIDs we can classify."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception as exc:
        eprint(f"warning: nvidia-smi GPU query failed ({exc})")
        return []

    gpus: list[GpuSnap] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(
                GpuSnap(
                    index=parts[0],
                    free_mib=int(float(parts[1])),
                    used_mib=int(float(parts[2])),
                    movable=[],
                )
            )
        except ValueError:
            continue

    by_idx = {g.index: g for g in gpus}
    try:
        pout = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,gpu_bus_id,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        # Map bus id / index via a second query — simpler: use pmon-style index query.
        _ = pout  # kept for future UUID mapping; index path below is enough
    except Exception:
        pass

    # Per-process VRAM with GPU index (nvidia-smi pmon lacks used mem; use query + index).
    try:
        # nvidia-smi --query-compute-apps does not include GPU index on all drivers;
        # fall back to parsing `nvidia-smi` compute apps table via --query with uuid map.
        uout = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        uuid_to_idx = {}
        for line in uout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[1]] = parts[0]
        aout = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        for line in aout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            uuid, pid_s, used_s = parts[0], parts[1], parts[2]
            idx = uuid_to_idx.get(uuid)
            if idx is None or idx not in by_idx:
                continue
            try:
                pid = int(pid_s)
                used = int(float(used_s))
            except ValueError:
                continue
            name = _cmdline(pid)
            if _is_movable(name):
                by_idx[idx].movable.append((pid, used, name))
    except Exception as exc:
        eprint(f"warning: nvidia-smi compute-apps query failed ({exc})")

    return gpus


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _is_movable(cmdline: str) -> bool:
    if not cmdline:
        return False
    if any(m in cmdline for m in _IMMOVABLE_MARKERS):
        return False
    return any(m in cmdline for m in _MOVABLE_CMDLINE_MARKERS)


def infer_need_mib() -> int:
    """Rough contiguous free-VRAM target for the upcoming DiT job."""
    for key in (
        "VIDEO_VRAM_NEED_MIB",
        "LTX_VRAM_NEED_MIB",
        "H3_VRAM_NEED_MIB",
        "WAN_VRAM_NEED_MIB",
    ):
        v = _env(key)
        if v.isdigit():
            return int(v)
    blob = " ".join(
        [
            _env("LTX_MODEL_TYPE"),
            _env("LTX_PROFILE"),
            _env("H3_MODEL_TYPE"),
            _env("WAN_PROFILE"),
            _env("WAN_MODEL_TYPE"),
        ]
    ).lower()
    if "ltx2" in blob or "22b" in blob:
        return 18000
    if "h3" in blob or "minimax" in blob:
        return 16000
    if "13b" in blob or "1.3b" in blob:
        return 14000
    if "2b" in blob:
        return 8000
    return int(_env("VIDEO_VRAM_NEED_MIB_DEFAULT", "14000") or "14000")


def _policy() -> str:
    return (_env("ZEROLLAMA_CUDA_POLICY") or "best_fit").lower().replace("-", "_")


def pick_cuda_device(
    *,
    need_mib: int | None = None,
    override_env: str = "",
    default: str = "0",
    migrate: bool = True,
    log: Callable[[str], None] | None = None,
) -> str:
    """Pick a GPU under best-fit / freest policy; optionally migrate irodori."""
    log = log or eprint
    if _env("CUDA_VISIBLE_DEVICES"):
        return _env("CUDA_VISIBLE_DEVICES").split(",")[0] or default
    if override_env and _env(override_env):
        return _env(override_env)

    need = need_mib if need_mib is not None else infer_need_mib()
    gpus = snapshot_gpus()
    if not gpus:
        return default

    policy = _policy()
    if policy == "freest":
        best = max(gpus, key=lambda g: g.free_mib)
        log(f"cuda policy=freest → GPU {best.index} ({best.free_mib} MiB free, need {need})")
        return best.index

    # 1) Best-fit among GPUs that already satisfy need (no migrate).
    fits = [g for g in gpus if g.free_mib >= need]
    if fits:
        # Smallest free that fits → leave the largest open chunk elsewhere.
        best = min(fits, key=lambda g: g.free_mib)
        log(
            f"cuda policy=best_fit → GPU {best.index} "
            f"({best.free_mib} MiB free ≥ need {need}; pack, keep larger chunk open)"
        )
        return best.index

    if not migrate or _env("ZEROLLAMA_CUDA_MIGRATE") in ("0", "false", "no", "off"):
        # Fall back to freest and hope mmgp survives — caller still gets a device.
        best = max(gpus, key=lambda g: g.free_mib)
        log(
            f"cuda policy=best_fit: no GPU has {need} MiB free; "
            f"migrate disabled → freest GPU {best.index} ({best.free_mib} MiB)"
        )
        return best.index

    # 2) Solvable contention: movable residents free enough headroom.
    candidates: list[tuple[int, GpuSnap, list[tuple[int, int, str]]]] = []
    for g in gpus:
        mov = g.movable
        reclaim = sum(u for _, u, _ in mov)
        if g.free_mib + reclaim >= need and mov:
            candidates.append((g.free_mib + reclaim, g, mov))
    if not candidates:
        best = max(gpus, key=lambda g: g.free_mib)
        log(
            f"cuda policy=best_fit: need {need} MiB; no solvable migrate "
            f"(immovable residents). Using freest GPU {best.index} ({best.free_mib} MiB)"
        )
        return best.index

    # Prefer the GPU that becomes the largest contiguous free after migrate.
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, target, movables = candidates[0]
    other = next((g.index for g in gpus if g.index != target.index), "0")
    log(
        f"cuda contention on GPU {target.index}: need {need} MiB, free {target.free_mib}; "
        f"migrating {len(movables)} movable pid(s) → GPU {other}"
    )
    if _migrate_movables(movables, dest_index=other, log=log):
        # Re-snapshot; best-fit again.
        time.sleep(1.0)
        gpus = snapshot_gpus()
        fits = [g for g in gpus if g.free_mib >= need]
        if fits:
            best = min(fits, key=lambda g: g.free_mib)
            log(f"cuda after migrate → GPU {best.index} ({best.free_mib} MiB free)")
            return best.index
        # Target should now be freest.
        if gpus:
            best = max(gpus, key=lambda g: g.free_mib)
            log(f"cuda after migrate (freest) → GPU {best.index} ({best.free_mib} MiB free)")
            return best.index

    best = max(gpus, key=lambda g: g.free_mib)
    log(f"cuda migrate failed; freest GPU {best.index} ({best.free_mib} MiB free)")
    return best.index


def _migrate_movables(
    movables: list[tuple[int, int, str]],
    *,
    dest_index: str,
    log: Callable[[str], None],
) -> bool:
    """Relocate irodori by rewriting its CUDA env file and signaling restart."""
    if not any("irodori" in name for _, _, name in movables):
        log("cuda migrate: no irodori among movables; skip")
        return False
    try:
        path = _IRODORI_CUDA_ENV
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"CUDA_VISIBLE_DEVICES={dest_index}\n")
        os.replace(tmp, path)
        log(f"cuda migrate: wrote {path} → CUDA_VISIBLE_DEVICES={dest_index}")
    except OSError as exc:
        log(f"cuda migrate: cannot write {_IRODORI_CUDA_ENV}: {exc}")
        return False

    # Same-uid SIGTERM; systemd Restart=on-failure reloads EnvironmentFile.
    killed = False
    for pid, _, name in movables:
        if "irodori" not in name:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"cuda migrate: SIGTERM irodori pid={pid}")
            killed = True
        except OSError as exc:
            log(f"cuda migrate: kill pid={pid}: {exc}")
    if not killed:
        return False

    # Wait until old VRAM releases (or timeout).
    deadline = time.time() + 45
    while time.time() < deadline:
        time.sleep(1.0)
        alive = False
        for pid, _, name in movables:
            if "irodori" not in name:
                continue
            if os.path.exists(f"/proc/{pid}"):
                alive = True
                break
        if not alive:
            return True
    log("cuda migrate: timed out waiting for irodori exit")
    return True  # env file is updated; placement may still improve


# Back-compat names used by wrappers.
def pick_freest_cuda_device(
    *,
    override_env: str = "",
    default: str = "0",
    log: Callable[[str], None] | None = None,
) -> str:
    prev = os.environ.get("ZEROLLAMA_CUDA_POLICY")
    os.environ["ZEROLLAMA_CUDA_POLICY"] = "freest"
    try:
        return pick_cuda_device(override_env=override_env, default=default, migrate=False, log=log)
    finally:
        if prev is None:
            os.environ.pop("ZEROLLAMA_CUDA_POLICY", None)
        else:
            os.environ["ZEROLLAMA_CUDA_POLICY"] = prev


def apply_freest_cuda_device(*, override_env: str = "", label: str = "video") -> str:
    """Deprecated name: applies best-fit (+ migrate) policy unless overridden."""
    return apply_cuda_device(override_env=override_env, label=label)


def apply_cuda_device(*, override_env: str = "", label: str = "video", need_mib: int | None = None) -> str:
    """Set CUDA_VISIBLE_DEVICES under the dual-GPU policy. Returns device index."""
    if _env("CUDA_VISIBLE_DEVICES"):
        return _env("CUDA_VISIBLE_DEVICES").split(",")[0] or "0"
    device = pick_cuda_device(override_env=override_env, need_mib=need_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    eprint(f"{label} using CUDA_VISIBLE_DEVICES={device}")
    return device
