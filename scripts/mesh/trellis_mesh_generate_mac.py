#!/usr/bin/env python3
"""TRELLIS.2 image → PBR GLB on Darwin (MPS + trellis-mac Metal backends).

Does not call trellis-mac generate.py. Uses their MPS env preamble and
CPU / fast_simplification bake sequence, then our room post-steps
(DINOv3 mirror, rembg stub, alphaMode, PNG).

Lab only. Source env-mac.sh first. Do not put this venv on the serve path.

Examples:
  source ~/.zerollama/third_party/trellis-mac/env-mac.sh
  python scripts/mesh/trellis_mesh_generate_mac.py \\
    --image /tmp/trellis-lab/still_mug_connected.png \\
    --out ~/.zerollama/third_party/trellis-mac/lab/out/trellis2-mps.glb

  # Closed crust from an existing GLB (no resample). Needs scikit-image.
  python scripts/mesh/trellis_mesh_generate_mac.py \\
    --from-glb ~/.zerollama/third_party/trellis-mac/lab/out/trellis2-mps-welded.glb \\
    --out ~/.zerollama/third_party/trellis-mac/lab/out/trellis2-mps-shell.glb
"""
from __future__ import annotations

# MPS env MUST be set before torch (or flex_gemm) is imported anywhere.
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

_TRELLIS_ROOT = os.environ.get(
    "TRELLIS_ROOT",
    os.path.join(os.path.expanduser("~"), ".zerollama", "third_party", "trellis-mac"),
)

try:
    import flex_gemm  # noqa: F401

    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError) as _e:
    print(
        f"error: flex_gemm required for Metal acceptance path ({_e})",
        file=sys.stderr,
    )
    raise SystemExit(2) from _e

# trellis-mac layout: TRELLIS.2 + stubs (stubs appended so pip o_voxel wins).
_sys_trellis2 = os.path.join(_TRELLIS_ROOT, "TRELLIS.2")
_sys_stubs = os.path.join(_TRELLIS_ROOT, "stubs")
if os.path.isdir(_sys_trellis2) and _sys_trellis2 not in sys.path:
    sys.path.insert(0, _sys_trellis2)
if os.path.isdir(_sys_stubs) and _sys_stubs not in sys.path:
    sys.path.append(_sys_stubs)

import argparse
import json
import time
from pathlib import Path

# Room helpers live in the CUDA wrapper module (no import-time CUDA).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from trellis_mesh_generate import (  # noqa: E402
    _alpha_mode_from_material,
    _apply_room_material,
    _image_has_useful_alpha,
    _install_dinov3_mirror,
    _install_rembg_stub_or_mirror,
)


def _progress(msg: str) -> None:
    print(f"PROGRESS: {msg}", flush=True)


def _dep_shas(root: Path) -> dict:
    out = {
        "trellis_mac_commit": os.environ.get("TRELLIS_MAC_COMMIT", ""),
        "mtlgemm": os.environ.get("TRELLIS_DEP_MTLGEMM", ""),
        "mtlmesh": os.environ.get("TRELLIS_DEP_MTLMESH", ""),
        "mtlbvh": os.environ.get("TRELLIS_DEP_MTLBVH", ""),
        "mtldiffrast": os.environ.get("TRELLIS_DEP_MTLDIFFRAST", ""),
        "trellis2_apple": os.environ.get("TRELLIS_DEP_TRELLIS2_APPLE", ""),
        "trellis2": os.environ.get("TRELLIS_DEP_TRELLIS2", ""),
    }
    import subprocess

    def _rev(rel: str) -> str:
        p = root / rel
        if not p.exists():
            return "missing"
        try:
            return subprocess.check_output(
                ["git", "-C", str(p), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            return "unknown"

    if not out["trellis_mac_commit"]:
        out["trellis_mac_commit"] = _rev(".")
    mapping = {
        "mtlgemm": "deps/mtlgemm",
        "mtlmesh": "deps/mtlmesh",
        "mtlbvh": "deps/mtlbvh",
        "mtldiffrast": "deps/mtldiffrast",
        "trellis2_apple": "deps/trellis2-apple",
        "trellis2": "TRELLIS.2",
    }
    for k, rel in mapping.items():
        if not out[k]:
            out[k] = _rev(rel)
    return out


def _require_metal_baker() -> dict:
    import o_voxel.postprocess as pp

    info = {
        "backend": getattr(pp, "_BACKEND", None),
        "has_dr": bool(getattr(pp, "_HAS_DR", False)),
        "has_flex_gemm": bool(getattr(pp, "_HAS_FLEX_GEMM", False)),
    }
    if info["backend"] != "metal" or not info["has_dr"] or not info["has_flex_gemm"]:
        msg = (
            "Metal baker required "
            f"(_BACKEND={info['backend']!r} _HAS_DR={info['has_dr']} "
            f"_HAS_FLEX_GEMM={info['has_flex_gemm']}). "
            "KDTree / conv_none are not the acceptance path."
        )
        print(f"error: {msg}", file=sys.stderr)
        raise RuntimeError(msg)
    return info


def _load_primary_mesh(glb_path: Path):
    import trimesh

    loaded = trimesh.load(str(glb_path), force="scene")
    geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    if not geoms:
        raise RuntimeError(f"no geometry in {glb_path}")
    return geoms[0], loaded


def _face_components(g) -> list:
    """Face-adjacency components. Their face counts sum to len(g.faces).

    trimesh.split() can count a face in more than one part on a non-manifold
    mesh, so component sizes from split() are not a partition.
    """
    import numpy as np
    from trimesh import graph

    n_faces = len(g.faces)
    if n_faces == 0:
        return []
    return list(
        graph.connected_components(
            g.face_adjacency,
            min_len=1,
            engine="scipy",
            nodes=np.arange(n_faces),
        )
    )


def _topology_from_mesh(g) -> dict:
    """Face-connected component stats for a room-mesh sanity check."""
    faces = int(len(g.faces))
    try:
        comps = _face_components(g)
        sizes = sorted((int(len(c)) for c in comps), reverse=True)
        n = len(sizes)
        largest = sizes[0] if sizes else 0
        frac = (largest / faces) if faces else 0.0
        if sum(sizes) != faces:
            raise RuntimeError(f"component faces {sum(sizes)} != mesh faces {faces}")
    except Exception as e:
        return {
            "error": str(e),
            "components": -1,
            "faces": faces,
            "largest_frac": 0.0,
            "watertight": bool(getattr(g, "is_watertight", False)),
        }
    return {
        "components": n,
        "faces": faces,
        "largest_faces": largest,
        "largest_frac": round(frac, 4),
        "watertight": bool(getattr(g, "is_watertight", False)),
        "winding_consistent": bool(getattr(g, "is_winding_consistent", False)),
        "top_component_faces": sizes[:8],
        "component_faces_sum": int(sum(sizes)),
    }


def _topology_report(glb_path: Path) -> dict:
    g, _ = _load_primary_mesh(glb_path)
    return _topology_from_mesh(g)


def _is_cube_floor(extent, *, thin: float, wide: float) -> bool:
    """Paper sheet that spans both long axes of the unit cube.

    min < thin and max > wide would also match a long baseboard. Require the
    two longer axes both > wide so only a near-full floor square matches.
    """
    axes = sorted(float(x) for x in extent)
    return axes[0] < thin and axes[1] > wide and axes[2] > wide


def _component_extents(g, face_idx) -> tuple:
    """Return (min_extent, max_extent, extent_xyz) for a face set."""
    import numpy as np

    faces = g.faces[np.asarray(face_idx, dtype=np.int64)]
    vids = np.unique(faces.reshape(-1))
    v = np.asarray(g.vertices)[vids]
    if len(v) == 0:
        return 0.0, 0.0, [0.0, 0.0, 0.0]
    ext = v.max(axis=0) - v.min(axis=0)
    return float(ext.min()), float(ext.max()), [round(float(x), 4) for x in ext]


def _position_weld_mesh(g, *, eps: float = 1e-5, uv_tol: float = 1e-3):
    """Merge vertices within eps by position. Keep one UV when a group disagrees.

    Returns (new_mesh, weld_info). UV disagreement at seams can smear textures;
    inspect only checks that PNG textures exist.
    """
    import numpy as np
    import trimesh
    from collections import defaultdict

    verts = np.asarray(g.vertices, dtype=np.float64)
    faces = np.asarray(g.faces, dtype=np.int64).copy()
    uv = None
    if getattr(g, "visual", None) is not None and getattr(g.visual, "uv", None) is not None:
        uv = np.asarray(g.visual.uv, dtype=np.float64)

    q = np.round(verts / eps).astype(np.int64)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, key in enumerate(map(tuple, q)):
        buckets[key].append(i)

    parent = np.arange(len(verts), dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    multi_groups = 0
    uv_disagree_groups = 0
    for ix in buckets.values():
        if len(ix) < 2:
            continue
        multi_groups += 1
        if uv is not None:
            u = uv[ix]
            if float(np.max(np.ptp(u, axis=0))) >= uv_tol:
                uv_disagree_groups += 1
        root = ix[0]
        for j in ix[1:]:
            union(root, j)

    roots = np.array([find(i) for i in range(len(verts))], dtype=np.int64)
    unique_roots, new_ids = np.unique(roots, return_inverse=True)
    new_verts = verts[unique_roots]
    new_faces = new_ids[faces]
    ok = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[ok]
    new_uv = uv[unique_roots] if uv is not None else None

    out = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    if new_uv is not None and getattr(g, "visual", None) is not None:
        try:
            from trimesh.visual.texture import TextureVisuals

            material = getattr(g.visual, "material", None)
            out.visual = TextureVisuals(uv=new_uv, material=material)
        except Exception:
            out.visual.uv = new_uv

    info = {
        "eps": eps,
        "uv_tol": uv_tol,
        "coincident_groups": multi_groups,
        "uv_disagree_groups": uv_disagree_groups,
        "verts_before": int(len(verts)),
        "verts_after": int(len(new_verts)),
        "faces_before": int(len(faces)),
        "faces_after": int(len(new_faces)),
        "degenerate_dropped": int((~ok).sum()),
    }
    return out, info


def _drop_cube_floor_sheets(
    g,
    *,
    thin: float = 0.02,
    wide: float = 0.9,
) -> tuple:
    """Drop face-connected components that are paper-thin AND cube-wide.

    A mug body can be a thin surface with a thick AABB. Drop a sheet only when
    both longer axes are > wide and the short axis is < thin, and a remaining
    component still has min extent >= thin. A lone flat object is kept.
    """
    import numpy as np

    n_faces = len(g.faces)
    if n_faces == 0:
        return g, {"dropped_sheets": 0, "dropped_faces": 0, "kept": [], "dropped": []}
    components = _face_components(g)
    keep = []
    dropped = []
    for c in components:
        mn, mx, ext = _component_extents(g, c)
        entry = {
            "faces": int(len(c)),
            "min_extent": round(mn, 4),
            "max_extent": round(mx, 4),
            "extent": ext,
        }
        if _is_cube_floor(ext, thin=thin, wide=wide):
            dropped.append(entry)
        else:
            keep.append((c, entry))
    if not dropped:
        return g, {
            "dropped_sheets": 0,
            "dropped_faces": 0,
            "thin": thin,
            "wide": wide,
            "kept": [e for _, e in keep],
            "dropped": [],
        }
    volumetric = [e for _, e in keep if e["min_extent"] >= thin]
    if not keep or not volumetric:
        # Never delete the whole mesh, or a flat object that has no solid piece beside it.
        return g, {
            "dropped_sheets": 0,
            "dropped_faces": 0,
            "thin": thin,
            "wide": wide,
            "kept": [e for _, e in keep],
            "dropped": dropped,
            "refused": "no_volumetric_component_left" if keep else "all_components_matched_floor_rule",
        }
    mask = np.zeros(n_faces, dtype=bool)
    for c, _ in keep:
        mask[np.asarray(c, dtype=np.int64)] = True
    g.update_faces(mask)
    g.remove_unreferenced_vertices()
    return g, {
        "dropped_sheets": len(dropped),
        "dropped_faces": int(sum(d["faces"] for d in dropped)),
        "thin": thin,
        "wide": wide,
        "kept": [e for _, e in keep],
        "dropped": dropped,
    }


def _prune_dust_components(
    glb_path: Path,
    *,
    min_faces: int,
    keep_largest: int,
    position_weld: bool = True,
    drop_cube_floors: bool = True,
) -> dict:
    """Weld coincident verts, drop cube-wide floor sheets, then face-count prune.

    Order: position weld (keep one UV at seams) → drop paper-thin AND cube-wide
    sheets → keep largest by face count. fill_holes does not weld components.
    """
    import numpy as np

    g, _scene = _load_primary_mesh(glb_path)
    before = _topology_from_mesh(g)
    weld_info = None
    sheets_info = None

    if position_weld and len(g.faces) > 0:
        g, weld_info = _position_weld_mesh(g)
        weld_info["components_before"] = before["components"]
        weld_info["components_after"] = _topology_from_mesh(g)["components"]

    if drop_cube_floors and len(g.faces) > 0:
        g, sheets_info = _drop_cube_floor_sheets(g)

    n_faces = len(g.faces)
    if n_faces == 0:
        return {
            "pruned": False,
            "min_faces": min_faces,
            "keep_largest": keep_largest,
            "before": before,
            "after": before,
            "weld": weld_info,
            "dropped_sheets": sheets_info,
        }
    components = _face_components(g)
    ranked = sorted(components, key=len, reverse=True)
    keep_idx = [c for c in ranked if len(c) >= min_faces]
    if keep_largest > 0:
        keep_idx = keep_idx[:keep_largest]
    if not keep_idx:
        keep_idx = [ranked[0]]
    face_pruned = len(keep_idx) != len(components)
    if face_pruned:
        mask = np.zeros(n_faces, dtype=bool)
        for c in keep_idx:
            mask[np.asarray(c, dtype=np.int64)] = True
        g.update_faces(mask)
        g.remove_unreferenced_vertices()

    g.export(str(glb_path))
    after = _topology_from_mesh(g)
    largest_mn = None
    largest_ext = None
    comps_now = _face_components(g)
    if comps_now:
        biggest = max(comps_now, key=len)
        largest_mn, _, largest_ext = _component_extents(g, biggest)
    changed = (
        face_pruned
        or (weld_info is not None and weld_info.get("verts_after") != weld_info.get("verts_before"))
        or (sheets_info is not None and int(sheets_info.get("dropped_sheets") or 0) > 0)
    )
    return {
        "pruned": bool(changed),
        "min_faces": min_faces,
        "keep_largest": keep_largest,
        "dropped_components": int(before["components"]) - int(after["components"]),
        "before": before,
        "after": after,
        "weld": weld_info,
        "dropped_sheets": sheets_info,
        "largest_min_extent": None if largest_mn is None else round(largest_mn, 4),
        "largest_extent": largest_ext,
        # Body means the largest piece is not a paper sheet. Extent check, not face fraction.
        "largest_is_body": largest_mn is not None and largest_mn >= 0.02,
    }


def _keep_largest_component(g):
    """Copy of the largest face-connected component."""
    import numpy as np

    comps = _face_components(g)
    if len(comps) <= 1:
        return g, len(comps)
    biggest = max(comps, key=len)
    mask = np.zeros(len(g.faces), dtype=bool)
    mask[np.asarray(biggest, dtype=np.int64)] = True
    out = g.copy()
    out.update_faces(mask)
    out.remove_unreferenced_vertices()
    return out, len(comps)


def write_watertight_shell(src: Path, dest: Path, *, divisions: int = 56) -> tuple[int, dict]:
    """Write a closed crust of the largest component. Does not replace src.

    Occupied voxels are marched. The result is a thin closed skin, not a
    filled solid: openings wider than the pitch stay as tunnels. Vertices are
    new, so UVs are copied from the nearest source vertex and the PNG is not
    re-baked. Needs scikit-image (trimesh marching cubes).
    """
    import numpy as np
    import trimesh

    g, _scene = _load_primary_mesh(src)
    body, ncomp = _keep_largest_component(g)
    ext = np.asarray(body.bounds[1] - body.bounds[0], dtype=np.float64)
    pitch = float(np.max(ext)) / float(divisions) if len(body.vertices) else 0.0
    note: dict = {
        "source": str(src),
        "out": str(dest),
        "method": "voxel_crust_marching_cubes",
        "divisions": int(divisions),
        "pitch": None if pitch == 0 else round(pitch, 6),
        "source_components": int(ncomp),
        "kept": "largest_component",
        "source_faces": int(len(body.faces)),
        "source_extent": [round(float(x), 4) for x in ext],
        "crust_not_solid": True,
        "pbr_rebake": False,
        "uv_transfer": "nearest_source_vertex",
    }
    if pitch <= 0 or len(body.faces) == 0:
        note["error"] = "empty mesh"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
        return 1, note
    try:
        from skimage.measure import marching_cubes
    except ImportError as e:
        note["error"] = (
            "scikit-image is required for the watertight crust "
            f"(uv pip install --python \"$TRELLIS_VENV/bin/python\" scikit-image): {e}"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
        print(f"error: {note['error']}", file=sys.stderr)
        return 1, note

    _progress(f"watertight_shell pitch={pitch:.5f} faces={len(body.faces)}")
    vox = body.voxelized(pitch)
    # Pad so the crust has an outside cell on every side.
    volume = np.pad(vox.matrix, 1).astype(np.float32)
    verts, faces, _normals, _values = marching_cubes(volume, level=0.5)
    origin = np.asarray(vox.translation, dtype=np.float64) - pitch
    shell = trimesh.Trimesh(
        vertices=origin + np.asarray(verts, dtype=np.float64) * pitch,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    sign_flipped = False
    if float(shell.volume) < 0:
        shell.invert()
        sign_flipped = True
    uv_ok = False
    visual = getattr(body, "visual", None)
    src_uv = getattr(visual, "uv", None) if visual is not None else None
    material = getattr(visual, "material", None) if visual is not None else None
    if src_uv is not None and material is not None and len(body.vertices) > 0:
        from scipy.spatial import cKDTree
        from trimesh.visual.texture import TextureVisuals

        _dist, idx = cKDTree(np.asarray(body.vertices, dtype=np.float64)).query(
            np.asarray(shell.vertices, dtype=np.float64),
            k=1,
        )
        shell.visual = TextureVisuals(
            uv=np.asarray(src_uv, dtype=np.float64)[np.asarray(idx, dtype=np.int64)],
            material=material,
        )
        uv_ok = True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shell.export(str(dest))
    loaded, _ = _load_primary_mesh(dest)
    topo = _topology_from_mesh(loaded)
    shell_ext = np.asarray(loaded.bounds[1] - loaded.bounds[0], dtype=np.float64)
    note.update(
        {
            "uv_transferred": uv_ok,
            "sign_flipped": sign_flipped,
            "faces": int(len(loaded.faces)),
            "volume": round(float(loaded.volume), 6),
            "shell_extent": [round(float(x), 4) for x in shell_ext],
            "topology": topo,
            "watertight": bool(topo.get("watertight")),
        }
    )
    from test_glb_inspect import inspect_glb as _rich_inspect

    note["glb_inspect"] = _rich_inspect(dest)
    dest.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    _progress(
        f"watertight_shell wrote {dest} watertight={note['watertight']} "
        f"faces={note['faces']} volume={note['volume']}"
    )
    if not note["watertight"]:
        print("error: voxel crust is not watertight", file=sys.stderr)
        return 1, note
    return 0, note


def _finish_line_ok(
    inspect: dict,
    *,
    alpha_mode: str,
    topology: dict,
    allow_fragmented: bool,
) -> list[str]:
    """Hard checks for the Mac lab finish line (not just --require-pbr)."""
    errs: list[str] = []
    if not inspect.get("baseColorTexture") or not inspect.get("metallicRoughnessTexture"):
        errs.append(f"missing PBR textures: {inspect}")
    mimes = inspect.get("imageMimeTypes") or []
    if not mimes:
        errs.append("glb_inspect missing imageMimeTypes (need PNG textures)")
    elif any(m != "image/png" for m in mimes):
        errs.append(f"expected image/png mime types, got {mimes}")
    modes = inspect.get("alphaModes") or []
    if not modes:
        errs.append("glb_inspect missing alphaModes")
    elif alpha_mode not in modes:
        errs.append(f"exported alphaMode={alpha_mode!r} not in glTF alphaModes={modes}")
    # Fragmentation gate: dust is not a room prop. After --keep-largest, several
    # face-connected components can remain (fill_holes does not weld them).
    comps = int(topology.get("components") or 0)
    frac = float(topology.get("largest_frac") or 0.0)
    faces = int(topology.get("faces") or 0)
    if not allow_fragmented:
        if comps < 0:
            errs.append(f"topology check failed: {topology}")
        elif comps > 50:
            errs.append(
                f"mesh too fragmented: {comps} components "
                f"(largest_frac={frac:.3f}). Raise --prune-min-faces / --keep-largest, "
                "use a photo of one object, or pass --allow-fragmented."
            )
        elif comps > 8 and frac < 0.4 and faces > 1000:
            errs.append(
                f"no dominant solid: {comps} components, largest_frac={frac:.3f}. "
                "Pass --allow-fragmented for lab debug."
            )
        elif frac < 0.1 and faces > 1000:
            errs.append(
                f"largest component is only {frac:.1%} of faces "
                f"({topology.get('largest_faces')}/{faces})."
            )
    return errs


def generate_one(
    *,
    image_path: Path,
    out_path: Path,
    pipeline=None,
    keep_pipeline: bool = False,
    resolution: int = 512,
    seed: int = 1,
    decimation: int = 100_000,
    texture_size: int = 1024,
    model: str = "microsoft/TRELLIS.2-4B",
    preprocess_image: bool = False,
    note_out: str = "",
    skip_bake: bool = False,
    allow_fragmented: bool = False,
    prune_min_faces: int = 100,
    keep_largest: int = 8,
    resident: bool = False,
) -> tuple[int, dict]:
    """Run one image→GLB. Optionally reuse a loaded pipeline (lab resident).

    Returns (exit_code, note). exit_code 0 on finish-line pass.
    """
    import torch
    from PIL import Image
    import o_voxel

    image_path = Path(image_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.is_file():
        return 2, {"error": f"image not found: {image_path}"}

    root = Path(_TRELLIS_ROOT)
    _progress(f"trellis_root={root}")

    if not torch.backends.mps.is_available():
        return 2, {"error": "MPS not available"}

    device = torch.device("mps")
    try:
        baker = _require_metal_baker()
    except RuntimeError as e:
        return 2, {"error": str(e)}
    _progress(f"device={device} baker={baker} resident={resident}")

    image = Image.open(image_path)
    has_alpha = _image_has_useful_alpha(image)
    do_preprocess = bool(preprocess_image) and not has_alpha
    dinov3 = _install_dinov3_mirror()
    rembg = _install_rembg_stub_or_mirror(stub=not do_preprocess)
    _progress(f"dinov3_model={dinov3} rembg={rembg} has_alpha={has_alpha} preprocess={do_preprocess}")

    pipeline_type = {512: "512", 1024: "1024_cascade"}[int(resolution)]
    dep_shas = _dep_shas(root)

    t0 = time.time()
    t_load = 0.0
    owns_pipeline = pipeline is None
    if pipeline is None:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        _progress(f"load_model={model} pipeline_type={pipeline_type}")
        t_load0 = time.time()
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model)
        pipeline.to(device)
        t_load = time.time() - t_load0
        _progress(f"load_sec={t_load:.1f}")
    else:
        _progress(f"reuse_pipeline pipeline_type={pipeline_type}")

    _progress(f"run seed={seed}")
    t_gen0 = time.time()
    try:
        outputs = pipeline.run(
            image,
            seed=int(seed),
            preprocess_image=do_preprocess,
            pipeline_type=pipeline_type,
        )
    except (IndexError, AssertionError) as e:
        msg = str(e)
        if any(sig in msg for sig in ("non-zero size", "BVH needs at least 8 triangles")):
            return 2, {
                "error": "empty mesh (likely Metal GPU watchdog)",
                "hint": "Retry once with MTL_CAPTURE_ENABLED=1; do not accept KDTree.",
            }
        raise
    t_gen = time.time() - t_gen0

    mesh = outputs[0] if isinstance(outputs, list) else outputs
    verts = mesh.vertices
    faces = mesh.faces
    n_faces = int(faces.shape[0]) if hasattr(faces, "shape") else 0
    if n_faces == 0 or (hasattr(verts, "shape") and verts.shape[0] == 0):
        return 2, {"error": "empty mesh after generate (Metal GPU watchdog)"}
    _progress(f"mesh faces={n_faces} gen_sec={t_gen:.1f}")

    if owns_pipeline and not keep_pipeline:
        del pipeline
        import gc

        gc.collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        _progress("pipeline_deleted")
        pipeline = None

    note = {
        "model": model,
        "image": str(image_path),
        "out": str(out_path),
        "seed": int(seed),
        "resolution": int(resolution),
        "pipeline_type": pipeline_type,
        "decimation_target": int(decimation),
        "texture_size": int(texture_size),
        "dinov3_model": dinov3,
        "rembg": rembg,
        "preprocess_image": do_preprocess,
        "device": "mps",
        "extractor": "mtldiffrast",
        "baker": baker,
        # Decode Mesh.fill_holes / remove_faces / simplify stay stubbed: trellis-mac
        # skips that path because it segfaulted on large decode meshes. Bounds
        # checks inside MtlMesh::fill_holes do not cover the preamble.
        "decode_hole_fill": "skipped",
        "bake_hole_fill": "not_run",
        "hole_fill": "decode_skipped;bake_not_run",
        "geometry_parity_with_astra": False,
        "sparse_conv_backend": os.environ.get("SPARSE_CONV_BACKEND", ""),
        "attn_backend": os.environ.get("ATTN_BACKEND", "sdpa"),
        "deps": dep_shas,
        "license_note": "mtldiffrast published MIT; lab checkout only, not vendored",
        "allow_fragmented": bool(allow_fragmented),
        "resident": bool(resident),
        "load_sec": round(t_load, 2),
        "generate_sec": round(t_gen, 2),
    }

    if skip_bake:
        note["elapsed_sec"] = round(time.time() - t0, 2)
        note_path = Path(note_out).expanduser() if note_out else out_path.with_suffix(".json")
        note_path.write_text(json.dumps(note, indent=2) + "\n")
        _progress("skip_bake")
        return 0, note

    import fast_simplification

    verts_t = mesh.vertices
    faces_t = mesh.faces
    verts_np = verts_t.detach().cpu().numpy() if hasattr(verts_t, "detach") else verts_t
    faces_np = faces_t.detach().cpu().numpy() if hasattr(faces_t, "detach") else faces_t
    target_pre = min(200_000, len(faces_np))
    pre_capped = len(faces_np) > target_pre
    if pre_capped:
        ratio = 1.0 - (target_pre / len(faces_np))
        _progress(f"pre_bake_simplify {len(faces_np):,} -> ~{target_pre:,}")
        simp_verts, simp_faces = fast_simplification.simplify(verts_np, faces_np, ratio)
        simp_verts_t = torch.from_numpy(simp_verts).float()
        simp_faces_t = torch.from_numpy(simp_faces.astype("int32"))
    else:
        simp_verts_t = verts_t.cpu() if hasattr(verts_t, "cpu") else verts_t
        simp_faces_t = faces_t.cpu() if hasattr(faces_t, "cpu") else faces_t

    _progress(f"bake decimation={decimation} texture={texture_size}")
    baker = _require_metal_baker()
    note["baker"] = baker
    t_bake0 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=simp_verts_t.cpu() if hasattr(simp_verts_t, "cpu") else simp_verts_t,
        faces=simp_faces_t.cpu() if hasattr(simp_faces_t, "cpu") else simp_faces_t,
        attr_volume=mesh.attrs.cpu() if hasattr(mesh.attrs, "cpu") else mesh.attrs,
        coords=mesh.coords.cpu() if hasattr(mesh.coords, "cpu") else mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(decimation),
        texture_size=int(texture_size),
        verbose=True,
    )
    alpha_mode = _alpha_mode_from_material(glb)
    _apply_room_material(glb, alpha_mode)
    glb.export(str(out_path), extension_webp=False)
    note["bake_sec"] = round(time.time() - t_bake0, 2)
    # to_glb default remesh=False: three CuMesh.fill_holes calls. Each returns
    # without new faces when no boundary loop has perimeter <= 0.03. This records
    # the calls, not that any hole was capped. Components are not welded.
    note["bake_pre_simplify_200k"] = bool(pre_capped)
    note["bake_hole_fill"] = (
        "to_glb non-remesh invokes CuMesh.fill_holes x3 "
        "(no-op unless a boundary loop has perimeter<=0.03); "
        + (
            "input fast_simplification-capped at 200k faces"
            if pre_capped
            else "input is the decode mesh"
        )
    )
    note["hole_fill"] = (
        "decode_skipped;bake_invokes_fill_holes_x3;"
        "noop_without_small_loop;does_not_weld_components"
    )

    from test_glb_inspect import inspect_glb as _rich_inspect

    prune_info = None
    if int(prune_min_faces) > 0 or int(keep_largest) > 0:
        prune_info = _prune_dust_components(
            out_path,
            min_faces=max(0, int(prune_min_faces)),
            keep_largest=max(0, int(keep_largest)),
        )
        weld = prune_info.get("weld") or {}
        sheets = prune_info.get("dropped_sheets") or {}
        _progress(
            "prune "
            f"dropped={prune_info.get('dropped_components', 0)} "
            f"components={prune_info.get('after', {}).get('components')} "
            f"largest_frac={prune_info.get('after', {}).get('largest_frac')} "
            f"weld_uv_disagree={weld.get('uv_disagree_groups')} "
            f"floor_sheets={sheets.get('dropped_sheets')}"
        )

    inspect = _rich_inspect(out_path)
    topology = _topology_report(out_path)
    _progress(f"topology components={topology.get('components')} largest_frac={topology.get('largest_frac')}")

    note["alphaMode"] = alpha_mode
    note["elapsed_sec"] = round(time.time() - t0, 2)
    note["glb_inspect"] = inspect
    note["topology"] = topology
    note["prune"] = prune_info
    note["prune_min_faces"] = int(prune_min_faces)
    note["keep_largest"] = int(keep_largest)

    note_path = Path(note_out).expanduser() if note_out else out_path.with_suffix(".json")
    note_path.write_text(json.dumps(note, indent=2) + "\n")
    _progress(f"wrote {out_path} alphaMode={alpha_mode} inspect={inspect}")

    errs = _finish_line_ok(
        inspect,
        alpha_mode=alpha_mode,
        topology=topology,
        allow_fragmented=bool(allow_fragmented),
    )
    if errs:
        note["errors"] = errs
        note_path.write_text(json.dumps(note, indent=2) + "\n")
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        return 1, note
    return 0, note


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", default="", help="Object PNG (prefer alpha / rembg'd)")
    ap.add_argument(
        "--from-glb",
        default="",
        help="Skip generation. Write a watertight crust of the largest component of this GLB",
    )
    ap.add_argument(
        "--shell-divisions",
        type=int,
        default=56,
        help="Voxel crust resolution along the longest axis (pitch = extent / this)",
    )
    ap.add_argument("--out", default="trellis2-mps.glb", help="Output GLB path")
    ap.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=(512, 1024),
        help="512 → pipeline 512; 1024 → 1024_cascade",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--decimation", type=int, default=100_000, help="Bake face target (~100k room)")
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--preprocess-image", action="store_true")
    ap.add_argument("--note-out", default="", help="JSON note path (default: beside GLB)")
    ap.add_argument("--skip-bake", action="store_true")
    ap.add_argument(
        "--allow-fragmented",
        action="store_true",
        help="Do not fail when the mesh has many face-connected components",
    )
    ap.add_argument(
        "--prune-min-faces",
        type=int,
        default=100,
        help="Drop face-connected components smaller than this after bake (0 disables size filter)",
    )
    ap.add_argument(
        "--keep-largest",
        type=int,
        default=8,
        help="After size filter, keep only the N largest components (0 = all that pass min-faces)",
    )
    args = ap.parse_args()
    if args.from_glb:
        src = Path(args.from_glb).expanduser()
        if args.out == "trellis2-mps.glb":
            dest = src.with_name(src.stem + "-shell.glb")
        else:
            dest = Path(args.out).expanduser()
        code, _note = write_watertight_shell(src, dest, divisions=args.shell_divisions)
        return code
    if not args.image:
        ap.error("--image is required unless --from-glb is set")
    code, _note = generate_one(
        image_path=Path(args.image),
        out_path=Path(args.out),
        resolution=args.resolution,
        seed=args.seed,
        decimation=args.decimation,
        texture_size=args.texture_size,
        model=args.model,
        preprocess_image=args.preprocess_image,
        note_out=args.note_out,
        skip_bake=args.skip_bake,
        allow_fragmented=args.allow_fragmented,
        prune_min_faces=args.prune_min_faces,
        keep_largest=args.keep_largest,
        keep_pipeline=False,
        resident=False,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
