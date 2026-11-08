#!/usr/bin/env python3
"""Lab text → shopping list → support placement → optional TRELLIS bake.

The chat model names props (prompt, size in meters, what they sit on). This
script places them: fit-inside uniform scale, pack along the support top, and
refuse out-of-room / off-support / overlapping slots. TRELLIS bakes are
unit-cube meshes; free model coordinates would plant the center in the floor.

Does not start serve. Chat is a client of OLLAMA_HOST (default :11434). Mesh
jobs go to the resident on 127.0.0.1:18091. Never binds 11434 or 8081.

Examples:
  python scripts/mesh/scene_plan_mac.py --self-test
  python scripts/mesh/scene_plan_mac.py \\
    --prompt "a desk with a mug on it" --plan-model llama3.2 --plan-only
  python scripts/mesh/scene_plan_mac.py \\
    --prompt "a desk with a mug on it" --plan-model llama3.2 --bake
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STILL_SH = REPO_ROOT / "scripts" / "mesh" / "mac_object_still.sh"
DEFAULT_LAB_HOST = "127.0.0.1"
DEFAULT_LAB_PORT = 18091
FLOOR_THICKNESS = 0.02
SURFACE_GAP = 0.05
AABB_EPS = 1e-4

PLAN_SYSTEM = """You plan a small indoor scene as a shopping list of separate props.
Return ONLY a JSON object with this shape:
{
  "room": {"width": <meters>, "depth": <meters>, "height": <meters>},
  "objects": [
    {
      "name": "<snake_case id>",
      "prompt": "<imagegen prompt for a single object on a plain white background>",
      "size": [<x meters>, <y meters>, <z meters>],
      "on": "floor" | "<name of another object>"
    }
  ]
}
Rules:
- At most the requested object count.
- Every object must have a unique name.
- size is the intended real-world AABB in meters (x=width, y=height, z=depth).
- on is "floor" or the name of a larger support that already appears in objects.
- Do not invent positions, yaw, or world coordinates.
- Do not include walls, floors, or rooms as objects.
- Prefer a few distinct props (furniture + items on them).
"""


def _progress(msg: str) -> None:
    print(f"PROGRESS: {msg}", flush=True)


def _vec3(v, *, label: str) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"{label} must be a list of 3 numbers")
    out = [float(x) for x in v]
    if any(x <= 0 for x in out):
        raise ValueError(f"{label} must be positive, got {out}")
    return out


def parse_plan(raw, *, max_objects: int) -> dict:
    """Validate a shopping-list plan. Raises ValueError on bad shape."""
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError("plan must be a JSON object or string")

    room = data.get("room")
    if not isinstance(room, dict):
        raise ValueError("missing room")
    width = float(room.get("width", 0))
    depth = float(room.get("depth", 0))
    height = float(room.get("height", 0))
    if width <= 0 or depth <= 0 or height <= 0:
        raise ValueError(f"room extents must be positive, got {width},{depth},{height}")

    objs = data.get("objects")
    if not isinstance(objs, list) or not objs:
        raise ValueError("objects must be a non-empty list")
    if len(objs) > max_objects:
        raise ValueError(f"too many objects: {len(objs)} > {max_objects}")

    seen: set[str] = set()
    clean = []
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            raise ValueError(f"objects[{i}] must be an object")
        name = str(o.get("name", "")).strip()
        prompt = str(o.get("prompt", "")).strip()
        on = str(o.get("on", "")).strip()
        if not name or not prompt or not on:
            raise ValueError(f"objects[{i}] needs name, prompt, and on")
        if name in seen:
            raise ValueError(f"duplicate name: {name}")
        if name == "floor":
            raise ValueError("name 'floor' is reserved")
        if "position" in o or "yaw" in o or "rotation" in o:
            raise ValueError(f"{name}: model must not set position/yaw")
        size = _vec3(o.get("size"), label=f"{name}.size")
        seen.add(name)
        clean.append({"name": name, "prompt": prompt, "size": size, "on": on})

    return {
        "room": {"width": width, "depth": depth, "height": height},
        "objects": clean,
    }


def _aabb_from_center_size(center: list[float], size: list[float]) -> list[list[float]]:
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    c0, c1, c2 = center
    return [
        [c0 - hx, c1 - hy, c2 - hz],
        [c0 + hx, c1 + hy, c2 + hz],
    ]


def _aabb_overlap(a: list[list[float]], b: list[list[float]]) -> bool:
    return (
        a[0][0] < b[1][0] - AABB_EPS
        and a[1][0] > b[0][0] + AABB_EPS
        and a[0][1] < b[1][1] - AABB_EPS
        and a[1][1] > b[0][1] + AABB_EPS
        and a[0][2] < b[1][2] - AABB_EPS
        and a[1][2] > b[0][2] + AABB_EPS
    )


def _inside_room(aabb: list[list[float]], room: dict) -> bool:
    w, d, h = room["width"], room["depth"], room["height"]
    return (
        aabb[0][0] >= -AABB_EPS
        and aabb[1][0] <= w + AABB_EPS
        and aabb[0][1] >= -AABB_EPS
        and aabb[1][1] <= h + AABB_EPS
        and aabb[0][2] >= -AABB_EPS
        and aabb[1][2] <= d + AABB_EPS
    )


def _topo_sort(objects: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (ordered, dropped). Supports must appear before dependents."""
    by_name = {o["name"]: o for o in objects}
    dropped: list[dict] = []
    remaining = {o["name"]: o for o in objects}
    ordered: list[dict] = []

    # Drop missing supports / self-supports first.
    for o in list(remaining.values()):
        on = o["on"]
        if on == "floor":
            continue
        if on == o["name"]:
            dropped.append({**o, "error": "on self"})
            remaining.pop(o["name"], None)
        elif on not in by_name:
            dropped.append({**o, "error": f"missing support {on!r}"})
            remaining.pop(o["name"], None)

    # Kahn: floor-supported first, then anything whose support is already ordered.
    while remaining:
        ready = [
            o
            for o in remaining.values()
            if o["on"] == "floor" or o["on"] in {x["name"] for x in ordered}
        ]
        if not ready:
            # Cycle among what is left.
            for o in remaining.values():
                dropped.append({**o, "error": "support cycle"})
            break
        # Stable: keep original relative order among ready items.
        ready_names = {o["name"] for o in ready}
        next_batch = [o for o in objects if o["name"] in ready_names and o["name"] in remaining]
        for o in next_batch:
            ordered.append(o)
            remaining.pop(o["name"], None)
    return ordered, dropped


def _fit_uniform_scale(slot: list[float], mesh_ext: list[float]) -> tuple[float, list[float]]:
    """Largest uniform scale that keeps the mesh inside the planned AABB."""
    if any(x <= 0 for x in mesh_ext):
        raise ValueError("empty mesh extent")
    scale = min(slot[i] / mesh_ext[i] for i in range(3))
    return scale, [mesh_ext[0] * scale, mesh_ext[1] * scale, mesh_ext[2] * scale]


def _on_support_top(aabb: list[list[float]], support_aabb: list[list[float]]) -> bool:
    """Object footprint stays on the support's top face (X/Z only)."""
    return (
        aabb[0][0] >= support_aabb[0][0] - AABB_EPS
        and aabb[1][0] <= support_aabb[1][0] + AABB_EPS
        and aabb[0][2] >= support_aabb[0][2] - AABB_EPS
        and aabb[1][2] <= support_aabb[1][2] + AABB_EPS
    )


def place_plan(
    plan: dict,
    *,
    mesh_extents: dict[str, list[float]] | None = None,
) -> dict:
    """Place objects by support contact. mesh_extents overrides planned size for scale."""
    room = plan["room"]
    ordered, dropped = _topo_sort(plan["objects"])
    placed: list[dict] = []
    occupied: list[tuple[str, list[list[float]]]] = []

    floor_size = [room["width"], FLOOR_THICKNESS, room["depth"]]
    floor_center = [room["width"] / 2.0, -FLOOR_THICKNESS / 2.0, room["depth"] / 2.0]
    floor_aabb = _aabb_from_center_size(floor_center, floor_size)
    floor_entry = {
        "name": "floor",
        "kind": "slab",
        "size": floor_size,
        "center": floor_center,
        "aabb": floor_aabb,
        "top_y": 0.0,
    }

    supports: dict[str, dict] = {"floor": floor_entry}
    # Next free X on each support top. Floor starts at the room's left edge + gap.
    surface_cursor_x: dict[str, float] = {"floor": SURFACE_GAP}

    for o in ordered:
        planned = list(o["size"])
        mesh_ext = None
        if mesh_extents and o["name"] in mesh_extents:
            mesh_ext = list(mesh_extents[o["name"]])
        # Planned size is the slot. Uniform scale fits the mesh inside that AABB.
        if mesh_ext is not None:
            try:
                scale, placed_size = _fit_uniform_scale(planned, mesh_ext)
            except ValueError:
                dropped.append({**o, "error": "empty mesh extent"})
                continue
        else:
            scale = 1.0
            placed_size = list(planned)

        support = supports.get(o["on"])
        if support is None:
            dropped.append({**o, "error": f"support {o['on']!r} was not placed"})
            continue

        top_y = float(support["top_y"])
        support_aabb = support["aabb"]
        if o["on"] not in surface_cursor_x:
            surface_cursor_x[o["on"]] = float(support_aabb[0][0]) + SURFACE_GAP

        cursor = surface_cursor_x[o["on"]]
        cx = cursor + placed_size[0] / 2.0
        if o["on"] == "floor":
            cz = room["depth"] / 2.0
        else:
            cz = float(support["center"][2])

        cy = top_y + placed_size[1] / 2.0
        center = [cx, cy, cz]
        aabb = _aabb_from_center_size(center, placed_size)

        if not _inside_room(aabb, room):
            dropped.append({**o, "error": "outside room", "aabb": aabb, "center": center})
            continue
        if o["on"] != "floor" and not _on_support_top(aabb, support_aabb):
            dropped.append({**o, "error": "off support", "aabb": aabb, "center": center})
            continue
        hit = None
        for other_name, other_aabb in occupied:
            if _aabb_overlap(aabb, other_aabb):
                hit = other_name
                break
        if hit is not None:
            dropped.append(
                {**o, "error": f"overlap with {hit}", "aabb": aabb, "center": center}
            )
            continue

        # Advance the surface cursor only after a successful place.
        surface_cursor_x[o["on"]] = float(aabb[1][0]) + SURFACE_GAP

        entry = {
            "name": o["name"],
            "prompt": o["prompt"],
            "on": o["on"],
            "planned_size": planned,
            "size": [round(x, 6) for x in placed_size],
            "scale": round(scale, 6),
            "center": [round(x, 6) for x in center],
            "aabb": [[round(x, 6) for x in aabb[0]], [round(x, 6) for x in aabb[1]]],
            "top_y": round(top_y + placed_size[1], 6),
            "yaw": 0.0,
        }
        if mesh_ext is not None:
            entry["mesh_extent"] = [round(x, 6) for x in mesh_ext]
        placed.append(entry)
        occupied.append((o["name"], aabb))
        supports[o["name"]] = entry

    return {
        "room": room,
        "floor": {
            "size": floor_size,
            "center": floor_center,
            "aabb": floor_aabb,
            "top_y": 0.0,
        },
        "placed": placed,
        "dropped": dropped,
    }


def chat_plan(prompt: str, *, host: str, model: str, max_objects: int) -> dict:
    base = host.rstrip("/")
    try:
        urllib.request.urlopen(urllib.request.Request(f"{base}/api/tags"), timeout=2)
    except Exception as e:
        raise RuntimeError(
            f"no serve at {host} — start chat, or pass a fixed plan JSON"
        ) from e

    user = (
        f"Scene request: {prompt}\n"
        f"Return at most {max_objects} objects. JSON only."
    )
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"chat HTTP {e.code}: {detail}") from e
    content = (payload.get("message") or {}).get("content")
    if not content:
        raise RuntimeError(f"empty chat reply: keys={list(payload.keys())}")
    return parse_plan(content, max_objects=max_objects)


def _mesh_extent(glb_path: Path) -> list[float]:
    import trimesh

    loaded = trimesh.load(str(glb_path), force="scene")
    geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    if not geoms:
        raise RuntimeError(f"no geometry in {glb_path}")
    g = geoms[0]
    ext = g.bounds[1] - g.bounds[0]
    return [float(ext[0]), float(ext[1]), float(ext[2])]


def _resident_job(
    *,
    host: str,
    port: int,
    image: Path,
    out: Path,
    seed: int,
    timeout: float = 1800.0,
) -> dict:
    if port in (11434, 8081):
        raise RuntimeError(f"refused production port {port}")
    payload = json.dumps(
        {"image": str(image), "out": str(out), "seed": int(seed)}
    ) + "\n"
    try:
        with socket.create_connection((host, port), timeout=5.0) as sock:
            sock.settimeout(timeout)
            sock.sendall(payload.encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except OSError as e:
        raise RuntimeError(
            f"TRELLIS resident not listening on {host}:{port} — "
            "start scripts/mesh/mac_mesh_lab.sh"
        ) from e
    if not buf:
        raise RuntimeError("empty reply from TRELLIS resident")
    line = buf.split(b"\n", 1)[0].decode("utf-8")
    return json.loads(line)


def bake_object(
    obj: dict,
    *,
    out_dir: Path,
    still_host: str,
    lab_host: str,
    lab_port: int,
    seed: int,
) -> dict:
    name = obj["name"]
    still = out_dir / f"{name}.png"
    glb = out_dir / f"{name}.glb"
    _progress(f"still {name}")
    cmd = [
        str(STILL_SH),
        "--prompt",
        obj["prompt"],
        "--out",
        str(still),
        "--host",
        still_host,
    ]
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        return {**obj, "error": f"still exit {r.returncode}"}
    _progress(f"mesh {name}")
    reply = _resident_job(
        host=lab_host, port=lab_port, image=still, out=glb, seed=seed
    )
    if not reply.get("ok"):
        return {
            **obj,
            "error": reply.get("error") or f"mesh exit {reply.get('exit_code')}",
            "still": str(still),
        }
    try:
        extent = _mesh_extent(glb)
    except Exception as e:
        return {**obj, "error": f"mesh extent: {e}", "still": str(still), "glb": str(glb)}
    return {
        **obj,
        "still": str(still),
        "glb": str(glb),
        "mesh_extent": extent,
        "bake_sec": reply.get("elapsed_sec"),
    }


def export_scene_glb(placement: dict, baked: dict[str, dict], dest: Path) -> None:
    import trimesh

    scene = trimesh.Scene()
    floor = placement["floor"]
    fs = floor["size"]
    fc = floor["center"]
    floor_mesh = trimesh.creation.box(extents=fs)
    floor_mesh.apply_translation(fc)
    scene.add_geometry(floor_mesh, node_name="floor", geom_name="floor")

    for p in placement["placed"]:
        b = baked.get(p["name"])
        if not b or not b.get("glb"):
            continue
        path = Path(b["glb"])
        loaded = trimesh.load(str(path), force="scene")
        geoms = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
        if not geoms:
            continue
        g = geoms[0].copy()
        # Center the mesh on its AABB, then scale, then move to placement center.
        bounds = g.bounds
        mid = (bounds[0] + bounds[1]) / 2.0
        g.apply_translation(-mid)
        scale = float(p["scale"])
        g.apply_scale(scale)
        g.apply_translation(p["center"])
        scene.add_geometry(g, node_name=p["name"], geom_name=p["name"])

    dest.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(dest))


def run_self_test() -> int:
    good = {
        "room": {"width": 4.0, "depth": 3.0, "height": 2.5},
        "objects": [
            {
                "name": "desk",
                "prompt": "a wooden desk",
                "size": [1.2, 0.75, 0.6],
                "on": "floor",
            },
            {
                "name": "mug",
                "prompt": "a red ceramic mug",
                "size": [0.09, 0.1, 0.09],
                "on": "desk",
            },
            {
                "name": "lamp",
                "prompt": "a desk lamp",
                "size": [0.15, 0.35, 0.15],
                "on": "desk",
            },
        ],
    }
    plan = parse_plan(good, max_objects=4)
    # Synthetic unit-cube extents (as if from TRELLIS).
    extents = {
        "desk": [1.0, 1.0, 1.0],
        "mug": [1.0, 1.0, 1.0],
        "lamp": [1.0, 1.0, 1.0],
    }
    placed = place_plan(plan, mesh_extents=extents)
    assert len(placed["placed"]) == 3, placed
    desk = next(p for p in placed["placed"] if p["name"] == "desk")
    mug = next(p for p in placed["placed"] if p["name"] == "mug")
    lamp = next(p for p in placed["placed"] if p["name"] == "lamp")
    desk_top = desk["aabb"][1][1]
    assert abs(desk_top - mug["aabb"][0][1]) < 1e-5, (desk_top, mug)
    assert abs(desk_top - lamp["aabb"][0][1]) < 1e-5, (desk_top, lamp)
    # Fit-inside slot: min(0.09/1, 0.1/1, 0.09/1)
    assert abs(mug["scale"] - 0.09) < 1e-6, mug["scale"]
    # Desk cube limited by depth 0.6, not longest 1.2
    assert abs(desk["scale"] - 0.6) < 1e-6, desk
    assert desk["size"] == [0.6, 0.6, 0.6], desk["size"]
    assert mug["center"][0] != lamp["center"][0], (mug["center"], lamp["center"])
    print(
        "ok: mug+lamp on desk; mug scale=",
        mug["scale"],
        "desk scale=",
        desk["scale"],
    )

    bad_missing = {
        "room": {"width": 3, "depth": 3, "height": 2.5},
        "objects": [
            {
                "name": "mug",
                "prompt": "mug",
                "size": [0.1, 0.1, 0.1],
                "on": "desk",
            }
        ],
    }
    miss = place_plan(parse_plan(bad_missing, max_objects=4))
    assert miss["placed"] == [] and miss["dropped"][0]["error"].startswith("missing"), miss
    print("ok: missing support dropped")

    # Cursor must not advance on a refused object: a too-wide first object
    # leaves room for a small second one at the start of the row.
    cramped = {
        "room": {"width": 1.0, "depth": 1.0, "height": 2.0},
        "objects": [
            {
                "name": "wide",
                "prompt": "wide",
                # 0.96 + SURFACE_GAP exceeds the 1.0 room; a following 0.2 must still fit.
                "size": [0.96, 0.4, 0.4],
                "on": "floor",
            },
            {
                "name": "small",
                "prompt": "small",
                "size": [0.2, 0.2, 0.2],
                "on": "floor",
            },
        ],
    }
    cramped_place = place_plan(parse_plan(cramped, max_objects=4))
    names = [p["name"] for p in cramped_place["placed"]]
    assert names == ["small"], cramped_place
    assert cramped_place["dropped"][0]["name"] == "wide"
    assert cramped_place["dropped"][0]["error"] == "outside room"
    print("ok: refused wide object does not steal the floor cursor")

    # Two large items on a small desk: second must fail off-support or overlap.
    crowded = {
        "room": {"width": 4.0, "depth": 3.0, "height": 2.5},
        "objects": [
            {
                "name": "desk",
                "prompt": "desk",
                "size": [0.5, 0.7, 0.5],
                "on": "floor",
            },
            {
                "name": "box_a",
                "prompt": "box",
                "size": [0.35, 0.2, 0.35],
                "on": "desk",
            },
            {
                "name": "box_b",
                "prompt": "box",
                "size": [0.35, 0.2, 0.35],
                "on": "desk",
            },
        ],
    }
    crowded_place = place_plan(parse_plan(crowded, max_objects=4))
    assert any(p["name"] == "box_a" for p in crowded_place["placed"]), crowded_place
    assert any(
        d["name"] == "box_b"
        and (d["error"].startswith("overlap") or d["error"] == "off support")
        for d in crowded_place["dropped"]
    ), crowded_place
    print(
        "ok: second desk item refused:",
        next(d["error"] for d in crowded_place["dropped"] if d["name"] == "box_b"),
    )

    try:
        parse_plan(
            {
                "room": {"width": 2, "depth": 2, "height": 2},
                "objects": [
                    {
                        "name": "a",
                        "prompt": "a",
                        "size": [1, 1, 1],
                        "on": "floor",
                        "position": [0, 0, 0],
                    }
                ],
            },
            max_objects=4,
        )
        print("error: expected position reject", file=sys.stderr)
        return 1
    except ValueError as e:
        print("ok: position from model rejected:", e)

    print("self-test passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--prompt", default="", help="Scene description for the planner")
    ap.add_argument("--plan-model", default="", help="Chat model tag on OLLAMA_HOST")
    ap.add_argument(
        "--plan-json",
        default="",
        help="Skip chat; load a shopping-list JSON file",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: ~/.zerollama/third_party/trellis-mac/lab/out/scene-<ts>)",
    )
    ap.add_argument("--max-objects", type=int, default=4)
    ap.add_argument("--plan-only", action="store_true", help="Write scene.json and stop")
    ap.add_argument(
        "--bake",
        action="store_true",
        help="Generate stills + TRELLIS GLBs and export scene.glb",
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Chat / imagegen host (client only)",
    )
    ap.add_argument(
        "--lab-host",
        default=os.environ.get("TRELLIS_LAB_HOST", DEFAULT_LAB_HOST),
    )
    ap.add_argument(
        "--lab-port",
        type=int,
        default=int(os.environ.get("TRELLIS_LAB_PORT", DEFAULT_LAB_PORT)),
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    if args.lab_port in (11434, 8081):
        print(f"error: refused production port {args.lab_port}", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path.home()
        / ".zerollama"
        / "third_party"
        / "trellis-mac"
        / "lab"
        / "out"
        / f"scene-{int(time.time())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plan_json:
        raw = json.loads(Path(args.plan_json).expanduser().read_text())
        plan = parse_plan(raw, max_objects=args.max_objects)
    else:
        if not args.prompt or not args.plan_model:
            ap.error("--prompt and --plan-model are required unless --plan-json or --self-test")
        try:
            plan = chat_plan(
                args.prompt,
                host=args.host,
                model=args.plan_model,
                max_objects=args.max_objects,
            )
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    note = {
        "prompt": args.prompt or None,
        "plan_model": args.plan_model or None,
        "plan": plan,
        "bake": bool(args.bake),
    }
    placement = place_plan(plan)
    note["placement"] = placement
    scene_json = out_dir / "scene.json"
    scene_json.write_text(json.dumps(note, indent=2) + "\n")
    _progress(
        f"wrote {scene_json} placed={len(placement['placed'])} "
        f"dropped={len(placement['dropped'])}"
    )

    if args.plan_only or not args.bake:
        if args.bake is False and not args.plan_only:
            # Default without --bake is plan+place only.
            pass
        return 0 if placement["placed"] else 1

    if not STILL_SH.is_file():
        print(f"error: missing {STILL_SH}", file=sys.stderr)
        return 2

    baked: dict[str, dict] = {}
    for p in placement["placed"]:
        obj = next(o for o in plan["objects"] if o["name"] == p["name"])
        try:
            result = bake_object(
                obj,
                out_dir=out_dir,
                still_host=args.host,
                lab_host=args.lab_host,
                lab_port=args.lab_port,
                seed=args.seed,
            )
        except Exception as e:
            result = {**obj, "error": str(e)}
        baked[p["name"]] = result
        if result.get("error"):
            _progress(f"bake failed {p['name']}: {result['error']}")

    # Re-place with measured mesh extents so scale matches the real GLB.
    extents = {
        name: b["mesh_extent"]
        for name, b in baked.items()
        if b.get("mesh_extent") and not b.get("error")
    }
    placement2 = place_plan(plan, mesh_extents=extents)
    # Drop objects whose bake failed even if the box placer would keep them.
    kept = []
    for p in placement2["placed"]:
        b = baked.get(p["name"]) or {}
        if b.get("error") or not b.get("glb"):
            placement2["dropped"].append(
                {**p, "error": b.get("error") or "bake missing"}
            )
            continue
        p["glb"] = b["glb"]
        p["still"] = b.get("still")
        kept.append(p)
    placement2["placed"] = kept

    note["baked"] = baked
    note["placement"] = placement2
    scene_json.write_text(json.dumps(note, indent=2) + "\n")

    scene_glb = out_dir / "scene.glb"
    try:
        export_scene_glb(placement2, baked, scene_glb)
        note["scene_glb"] = str(scene_glb)
        scene_json.write_text(json.dumps(note, indent=2) + "\n")
        _progress(f"wrote {scene_glb} objects={len(kept)}")
    except Exception as e:
        print(f"error: scene export failed: {e}", file=sys.stderr)
        return 1

    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
