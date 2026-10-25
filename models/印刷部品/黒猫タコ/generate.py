#!/usr/bin/env python3
"""黒猫タコの外装を生成する。

おばけねこの本体と足を土台にして、外側に耳・ほっぺ・しっぽ受け・タコ足を足す。
画面窓、ソケット室、ヒンジの通り道、足のはめあいは元のSTLのまま。
共通のヒンジと、おばけねこのソケットをそのまま使う。

再生成（印刷だけなら不要）:
  python3 -m venv /tmp/cadenv
  /tmp/cadenv/bin/pip install trimesh numpy manifold3d
  /tmp/cadenv/bin/python models/印刷部品/黒猫タコ/generate.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import trimesh
from trimesh import boolean as meshbool

ROOT = Path(__file__).resolve().parents[3]
SRC_BODY = ROOT / "models" / "印刷部品" / "おばけねこ" / "本体.stl"
SRC_FEET = ROOT / "models" / "印刷部品" / "おばけねこ" / "足.stl"
SRC_SOCK = ROOT / "models" / "印刷部品" / "おばけねこ" / "ソケット.stl"
SRC_HINGE = ROOT / "models" / "印刷部品" / "共通" / "ヒンジ.stl"
OUT = Path(__file__).resolve().parent

WINDOW_CENTER_Z = (0.8 + 25.0) / 2.0
FEET_BOTTOM_Z = -30.0
TAIL_HOLE_DIA = 8.4


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero vector")
    return v / n


def first_hit(mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray):
    direction = unit(direction)
    origin = np.asarray(origin, dtype=float)
    locs, _ray, tris = mesh.ray.intersects_location(
        origin.reshape(1, 3), direction.reshape(1, 3)
    )
    if len(locs) == 0:
        return None, None
    t = (locs - origin) @ direction
    ok = t > 1e-4
    if not np.any(ok):
        return None, None
    k = int(np.argmin(np.where(ok, t, 1e9)))
    return locs[k], int(tris[k])


def align_z_to(direction: np.ndarray) -> np.ndarray:
    new_z = unit(direction)
    up = np.array([0.0, 0.0, 1.0])
    new_x = np.cross(up, new_z)
    if np.linalg.norm(new_x) < 1e-6:
        new_x = np.cross(np.array([0.0, 1.0, 0.0]), new_z)
    new_x = unit(new_x)
    new_y = np.cross(new_z, new_x)
    rot = np.eye(4)
    rot[:3, 0] = new_x
    rot[:3, 1] = new_y
    rot[:3, 2] = new_z
    return rot


def align_x_to(direction: np.ndarray) -> np.ndarray:
    new_x = unit(direction)
    up = np.array([0.0, 0.0, 1.0])
    new_z = up - new_x * np.dot(up, new_x)
    if np.linalg.norm(new_z) < 1e-6:
        new_z = np.array([0.0, 1.0, 0.0])
    new_z = unit(new_z)
    new_y = np.cross(new_z, new_x)
    rot = np.eye(4)
    rot[:3, 0] = new_x
    rot[:3, 1] = new_y
    rot[:3, 2] = new_z
    return rot


def seal(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def union_all(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    acc = meshes[0]
    for i, part in enumerate(meshes[1:], start=1):
        t0 = time.time()
        acc = meshbool.union([acc, part], engine="manifold")
        print(
            f"  union {i}/{len(meshes) - 1}: faces {len(acc.faces)} "
            f"watertight {acc.is_watertight} vol {acc.volume:.0f} "
            f"({time.time() - t0:.1f}s)"
        )
    return seal(acc)


def difference(base: trimesh.Trimesh, cut: trimesh.Trimesh) -> trimesh.Trimesh:
    return seal(meshbool.difference([base, cut], engine="manifold"))


def truncated_cone(r_base: float, r_tip: float, height: float, sections: int = 28) -> trimesh.Trimesh:
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    bot = np.c_[r_base * np.cos(angles), r_base * np.sin(angles), np.zeros(sections)]
    top = np.c_[r_tip * np.cos(angles), r_tip * np.sin(angles), np.full(sections, height)]
    verts = np.vstack([bot, top, [[0.0, 0.0, 0.0]], [[0.0, 0.0, height]]])
    n = sections
    faces = []
    cb, ct = 2 * n, 2 * n + 1
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [cb, j, i], [ct, n + i, n + j]]
    return seal(trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True))


def d_profile(width: float, height: float, arc_steps: int = 16) -> np.ndarray:
    """Flat-bottom half-ellipse. Local z=0 is the bottom. `height` is the total height."""
    half_w = width / 2.0
    pts = [(-half_w, 0.0), (half_w, 0.0)]
    for a in np.linspace(0.0, math.pi, arc_steps)[1:-1]:
        pts.append((half_w * math.cos(a), height * math.sin(a)))
    return np.array(pts)


def loft_along(stations: list[tuple[np.ndarray, float, float]], z0: float) -> trimesh.Trimesh:
    """Loft half-ellipses. stations are (xy point, width, height). Bottoms lie on z=z0."""
    frames = []
    pts = [np.asarray(p, dtype=float) for p, _w, _h in stations]
    for i, ((p, width, height), _xy) in enumerate(zip(stations, pts)):
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == len(pts) - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]
        tangent = unit(np.array([tangent[0], tangent[1], 0.0]))
        across = unit(np.cross(np.array([0.0, 0.0, 1.0]), tangent))
        ring = []
        for y, z in d_profile(width, height):
            ring.append([p[0] + across[0] * y, p[1] + across[1] * y, z0 + z])
        frames.append(ring)
    nprof = len(frames[0])
    verts = np.array(frames, dtype=float).reshape(-1, 3)
    faces = []
    ns = len(frames)
    for s in range(ns - 1):
        for i in range(nprof):
            j = (i + 1) % nprof
            a = s * nprof + i
            b = s * nprof + j
            c = (s + 1) * nprof + j
            d = (s + 1) * nprof + i
            faces += [[a, b, c], [a, c, d]]
    for i in range(1, nprof - 1):
        faces.append([0, i, i + 1])
    last = (ns - 1) * nprof
    for i in range(1, nprof - 1):
        faces.append([last, last + i + 1, last + i])
    mesh = seal(trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True))
    if not mesh.is_watertight or mesh.volume <= 0:
        raise RuntimeError(f"loft failed vol={mesh.volume} watertight={mesh.is_watertight}")
    return mesh


def cylinder_x(radius: float, length: float, sections: int = 24) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, [0, 1, 0]))
    mesh.apply_translation([length / 2.0, 0.0, 0.0])
    return seal(mesh)


def subtract_halfspace(mesh: trimesh.Trimesh, point: np.ndarray, keep_normal: np.ndarray) -> trimesh.Trimesh:
    keep_normal = unit(keep_normal)
    box = trimesh.creation.box(extents=(220, 220, 220))
    box.apply_transform(align_z_to(-keep_normal))
    box.apply_translation(np.asarray(point, dtype=float) - keep_normal * (110.0 - 0.15))
    return difference(mesh, box)


def ellipsoid(radii: tuple[float, float, float], center: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.vertices *= np.asarray(radii, dtype=float)
    mesh.apply_translation(center)
    return seal(mesh)


def make_ear(body: trimesh.Trimesh, sign: float) -> trimesh.Trimesh:
    origin = np.array([sign * 19.4, -4.2, 90.0])
    hit, _tri = first_hit(body, origin, np.array([0.0, 0.0, -1.0]))
    if hit is None:
        raise RuntimeError(f"ear ray missed sign={sign}")
    # Upright enough that the sides print without support. The flat base is buried.
    axis = unit([sign * 0.20, -0.08, 0.976])
    sink = 18.0
    height = 44.0
    base = hit - axis * sink
    ear = truncated_cone(8.8, 1.05, height, sections=28)
    ear.apply_transform(align_z_to(axis))
    ear.apply_translation(base)
    print(f"  ear sign {sign:+.0f} hit {np.round(hit, 1)} tip {np.round(base + axis * height, 1)}")
    return ear


def make_cheek(body: trimesh.Trimesh, sign: float):
    origin = np.array([sign * 24.0, -55.0, 13.5])
    hit, tri = first_hit(body, origin, np.array([0.0, 1.0, 0.0]))
    if hit is None:
        raise RuntimeError(f"cheek ray missed sign={sign}")
    outward = np.array(body.face_normals[tri], dtype=float)
    if np.dot(outward, np.array([0.0, -1.0, 0.0])) < 0:
        outward = -outward
    outward = unit(outward)
    radii = (8.2, 6.6, 11.0)
    center = hit + outward * (radii[1] - 1.3) + np.array([sign * 1.2, 0.0, 0.4])
    cheek = ellipsoid(radii, center)
    # About 53° from horizontal, so the underside does not need support.
    cut_point = center + np.array([sign * 0.4, -0.4, -1.6])
    keep = unit([sign * -0.12, 0.78, 0.61])
    cheek = subtract_halfspace(cheek, cut_point, keep)
    print(f"  cheek sign {sign:+.0f} hit {np.round(hit, 1)} center {np.round(center, 1)} vol {cheek.volume:.0f}")
    return cheek, center


def make_whisker_holes(cheeks: list[tuple[float, trimesh.Trimesh, np.ndarray]]) -> list[trimesh.Trimesh]:
    holes = []
    radius = 1.55  # 3.1 mm. Whisker peg is 2.3 mm wide.
    depth = 5.0
    for sign, cheek, center in cheeks:
        for xoff, zoff in ((sign * 0.4, -4.6), (sign * 3.0, 0.5), (sign * 0.6, 5.6)):
            origin = np.array([center[0] + xoff, -48.0, center[2] + zoff])
            hit, tri = first_hit(cheek, origin, np.array([0.0, 1.0, 0.0]))
            if hit is None:
                raise RuntimeError(f"whisker ray missed sign={sign} zoff={zoff}")
            normal = np.array(cheek.face_normals[tri], dtype=float)
            if np.dot(normal, np.array([0.0, -1.0, 0.0])) < 0:
                normal = -normal
            direction = unit(normal + np.array([sign * 0.42, -0.12, zoff * 0.025]))
            hole = cylinder_x(radius, depth + 2.2, sections=18)
            hole.apply_transform(align_x_to(direction))
            hole.apply_translation(hit - direction * depth)
            holes.append(hole)
            print(f"  whisker aim sign {sign:+.0f} hit {np.round(hit, 1)}")
    return holes


def make_tail_boss(body: trimesh.Trimesh):
    origin = np.array([-62.0, 30.0, 16.0])
    hit, _tri = first_hit(body, origin, unit([1.0, -0.42, 0.05]))
    if hit is None:
        raise RuntimeError("tail boss ray missed")
    # 55° above horizontal, toward the viewer's left and the rear.
    axis = unit([-0.72, 0.48, 1.45])
    boss = cylinder_x(8.0, 16.0, sections=28)
    boss.apply_transform(align_x_to(axis))
    boss.apply_translation(hit - axis * 4.0)
    print(f"  tail boss hit {np.round(hit, 1)} axis {np.round(axis, 3)}")
    return boss, hit, axis


def make_tail_hole(hit: np.ndarray, axis: np.ndarray) -> trimesh.Trimesh:
    length = 11.0
    hole = cylinder_x(TAIL_HOLE_DIA / 2.0, length, sections=28)
    hole.apply_transform(align_x_to(axis))
    outer_end = hit - axis * 4.0 + axis * 16.0
    hole.apply_translation(outer_end - axis * (length - 1.2))
    return hole


def body_clear_radius(body: trimesh.Trimesh, angle_deg: float, wedge_deg: float = 14.0) -> float:
    ang = math.radians(angle_deg)
    center = np.array([0.0, 5.0])
    xy = body.vertices[:, :2] - center
    rad = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    delta = (theta - ang + np.pi) % (2 * np.pi) - np.pi
    sel = np.abs(delta) < math.radians(wedge_deg)
    if not np.any(sel):
        raise RuntimeError(f"no body verts near {angle_deg}")
    return float(rad[sel].max())


def make_tentacle(angle_deg: float, length: float, curl: float, w0: float, w1: float, body_r: float) -> trimesh.Trimesh:
    """Tab under the skirt, then a thick tentacle only once it is outside the shell."""
    ang = math.radians(angle_deg)
    u = np.array([math.cos(ang), math.sin(ang)])
    side = np.array([-u[1], u[0]])
    center = np.array([0.0, 5.0])
    start_r = 15.0
    escape_r = body_r + 1.15
    rise_r = body_r + 8.0
    tip_r = body_r + length
    stations = []
    for i in range(10):
        dist = start_r + (tip_r - start_r) * (i / 9.0)
        if dist < escape_r:
            height, lateral, width = 3.30, 0.0, max(w0, 16.0)
        elif dist < rise_r:
            t = (dist - escape_r) / (rise_r - escape_r)
            height = 3.30 + t * (12.6 - 3.30)
            lateral = curl * 3.0 * t * t
            width = w0 + (w1 - w0) * 0.15 * t
        else:
            t = (dist - rise_r) / max(tip_r - rise_r, 1.0)
            height = 12.6 + t * (7.8 - 12.6)
            lateral = curl * (3.0 + 11.0 * t)
            width = w0 + (w1 - w0) * (0.15 + 0.85 * t)
        stations.append((center + u * dist + side * lateral, width, height))
    mesh = loft_along(stations, FEET_BOTTOM_Z)
    if mesh.volume < 100:
        raise RuntimeError(f"tentacle {angle_deg} too small")
    return mesh


def tentacle_specs():
    # angle (0=+X, 90=rear, -90=front), length past the skirt, curl, root width, tip width
    return [
        (-150, 34, 1.0, 16.5, 10.0),
        (-112, 42, -1.0, 18.0, 11.0),
        (-74, 40, 1.0, 17.0, 10.5),
        (-36, 36, -1.0, 16.5, 10.0),
        (8, 32, 1.0, 16.0, 9.5),
        (62, 34, -1.0, 17.0, 10.5),
        (112, 40, 1.0, 18.0, 11.0),
        (158, 33, -1.0, 16.5, 10.0),
    ]


def make_base(feet: trimesh.Trimesh, body: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = []
    for i, comp in enumerate(feet.split(only_watertight=False)):
        if comp.is_watertight and comp.volume > 1.0 and len(comp.faces) > 20:
            parts.append(seal(comp))
            print(f"  feet component {i} vol {comp.volume:.0f}")
    feet_solid = parts[0] if len(parts) == 1 else meshbool.union(parts, engine="manifold")
    tentacles = []
    for spec in tentacle_specs():
        body_r = body_clear_radius(body, spec[0])
        tentacle = make_tentacle(*spec, body_r=body_r)
        overlap = meshbool.intersection([tentacle, feet_solid], engine="manifold")
        iv = 0.0 if overlap is None else float(overlap.volume)
        print(f"  tentacle {spec[0]} body_r {body_r:.1f} vol {tentacle.volume:.0f} feet-overlap {iv:.0f}")
        if iv < 40:
            raise RuntimeError(f"tentacle {spec[0]} does not join the feet")
        tentacles.append(tentacle)
    print("union feet + tentacles")
    return union_all(parts + tentacles)


def make_whiskers() -> trimesh.Trimesh:
    peg_w, peg_h, peg_len = 2.30, 2.05, 4.4
    pieces = []
    for i in range(6):
        stations = [
            (np.array([0.0, 0.0]), peg_w, peg_h),
            (np.array([peg_len, 0.0]), peg_w, peg_h),
            (np.array([peg_len + 0.5, 0.15]), 3.4, 3.05),
            (np.array([peg_len + 8.0, 0.8]), 3.2, 2.85),
            (np.array([peg_len + 16.0, 1.6]), 2.6, 2.35),
            (np.array([peg_len + 23.0, 2.4]), 2.0, 1.8),
        ]
        whisker = loft_along(stations, 0.0)
        whisker.apply_translation([0.0, i * 12.0, 0.0])
        pieces.append(whisker)
    return seal(trimesh.util.concatenate(pieces))


def make_tail() -> trimesh.Trimesh:
    peg_w, peg_h = 4.2, 3.7
    ctrl = [
        (-8.0, 0.0, peg_w, peg_h),
        (-0.2, 0.0, peg_w, peg_h),
        (1.5, 0.3, 11.0, 9.4),
        (16.0, 2.0, 10.2, 8.8),
        (32.0, 8.0, 9.0, 7.8),
        (48.0, 18.0, 7.4, 6.5),
        (62.0, 30.0, 6.0, 5.3),
        (72.0, 40.0, 5.0, 4.4),
    ]
    stations = [(np.array([c[0], c[1]]), c[2], c[3]) for c in ctrl]
    return loft_along(stations, 0.0)


def overhang_area(mesh: trimesh.Trimesh, limit_deg: float = 48.0) -> float:
    normals = mesh.face_normals
    down = np.maximum(-normals[:, 2], 0.0)
    horiz = np.linalg.norm(normals[:, :2], axis=1)
    slope = np.degrees(np.arctan2(horiz, np.maximum(down, 1e-9)))
    bad = (normals[:, 2] < -0.05) & (slope < limit_deg)
    return float(mesh.area_faces[bad].sum())


def export_stl(mesh: trimesh.Trimesh, path: Path) -> None:
    mesh.export(path)
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"wrote {path.name} faces {len(mesh.faces)} ext {np.round(ext, 1)}")


def main() -> None:
    print("load")
    body = load_mesh(SRC_BODY)
    feet = load_mesh(SRC_FEET)
    sock = load_mesh(SRC_SOCK)
    hinge = load_mesh(SRC_HINGE)

    print("ears / cheeks / tail boss")
    ears = [make_ear(body, s) for s in (-1.0, 1.0)]
    cheek_info = []
    cheeks = []
    for sign in (-1.0, 1.0):
        cheek, center = make_cheek(body, sign)
        cheeks.append(cheek)
        cheek_info.append((sign, cheek, center))
    boss, hit, axis = make_tail_boss(body)

    print("union head")
    head = union_all([body, *ears, *cheeks, boss])
    print("drill")
    before = head.volume
    for i, hole in enumerate(make_whisker_holes(cheek_info)):
        head = difference(head, hole)
        dropped = before - head.volume
        print(f"  whisker hole {i} removed {dropped:.0f} mm^3")
        if dropped < 20:
            raise SystemExit(f"whisker hole {i} did not open ({dropped:.1f} mm^3)")
        before = head.volume
    head = difference(head, make_tail_hole(hit, axis))
    print(f"head watertight {head.is_watertight} vol {head.volume:.0f}")

    print("base")
    base = make_base(feet, body)
    whiskers = make_whiskers()
    tail = make_tail()

    print("--- checks ---")
    origin = np.array([0.0, -70.0, WINDOW_CENTER_Z])
    loc_b, _ = first_hit(body, origin, np.array([0.0, 1.0, 0.0]))
    loc_h, _ = first_hit(head, origin, np.array([0.0, 1.0, 0.0]))
    print(f"window Y body {None if loc_b is None else round(float(loc_b[1]), 2)} head {None if loc_h is None else round(float(loc_h[1]), 2)}")

    inside_sock = int(head.contains(sock.vertices[::2]).sum())
    print(f"socket samples inside head {inside_sock}")
    hinge_body = int(body.contains(hinge.vertices).sum())
    hinge_head = int(head.contains(hinge.vertices).sum())
    print(f"hinge verts inside body {hinge_body} head {hinge_head}")

    rim = body.vertices[body.vertices[:, 2] < -8.0][::3]
    new_hits = int((base.contains(rim) & ~feet.contains(rim)).sum())
    print(f"body rim newly inside base {new_hits}/{len(rim)}")

    for name, mesh in (("head", head), ("base", base), ("tail", tail), ("whiskers", whiskers)):
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(
            f"{name}: watertight {mesh.is_watertight} vol {mesh.volume:.0f} "
            f"ext {np.round(ext, 1)} overhang<48 {overhang_area(mesh):.0f}"
        )
        if max(ext) > 176:
            raise SystemExit(f"{name} does not fit the A1 mini bed {ext}")

    peg = tail.vertices[tail.vertices[:, 0] < -0.2]
    peg_r = float(np.linalg.norm(peg[:, 1:] - peg[:, 1:].mean(axis=0), axis=1).max())
    print(f"peg enclosing radius {peg_r:.2f} hole radius {TAIL_HOLE_DIA / 2:.2f}")

    if loc_h is None or loc_h[1] < 15:
        raise SystemExit("screen window looks blocked")
    if inside_sock:
        raise SystemExit("socket collides with the head")
    if hinge_head > hinge_body + 5:
        raise SystemExit("hinge clearance got worse")
    if new_hits > 8:
        raise SystemExit("tentacles collide with the body")
    if peg_r > TAIL_HOLE_DIA / 2.0 - 0.45:
        raise SystemExit("tail peg does not fit the hole")
    for mesh, name in ((head, "head"), (base, "base"), (tail, "tail"), (whiskers, "whiskers")):
        if not mesh.is_watertight:
            raise SystemExit(f"{name} is not watertight")

    export_stl(head, OUT / "頭.stl")
    export_stl(base, OUT / "タコ足.stl")
    export_stl(tail, OUT / "しっぽ.stl")
    export_stl(whiskers, OUT / "ひげ.stl")
    print("done")


if __name__ == "__main__":
    main()
