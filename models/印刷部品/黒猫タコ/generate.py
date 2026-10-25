#!/usr/bin/env python3
"""黒猫タコ。SG90 が1個で、タコ足と尻尾を一緒にヨーイングする。

頭は固定。サーボの胴は頭に固定し、ホーンが「タコ足と尻尾」1部品を振る。
参考写真の簡略版。尻尾の巻きはサポートを減らすために浅くしてある。

  python3 generate.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, OpType, set_circular_segments
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent

# 組み立て座標。Z上、+Yが顔、+Xが右。サーボ軸はZ軸。
Z_BASE = 42.0
HEAD_R = 34.0
CUT = math.radians(52.0)
Z_CENTER = Z_BASE + HEAD_R * math.cos(CUT)
HUB_TOP = Z_BASE - 2.2

# SG90。軸は胴の長手で端に寄る。長い側を後ろ（-Y）。
BODY_X = 13.6
BODY_Y0 = -19.2
BODY_Y1 = 8.2
TAB_Y0 = -23.8
TAB_Y1 = 12.4
TAB_Z0 = 1.0
TAB_Z1 = 5.2
COLLAR_H = 6.3
COLLAR_R = 7.2
BODY_Z = 24.6
FLOOR = Z_BASE + COLLAR_H

# 尻尾は左後ろ。±32°振ってもサーボ室に入らない方位。
TAIL_AZ = math.radians(180.0 + 58.0)  # 数学角。180°が -X、+90°が +Y
# 180+58 = 238°: x=cos(238)<0, y=sin(238)<0 → 左後ろ。


def az_dir(extra: float = 0.0):
    a = TAIL_AZ + extra
    return np.array([math.cos(a), math.sin(a), 0.0])


def add(parts):
    parts = [p for p in parts if p is not None and not p.is_empty()]
    if not parts:
        raise RuntimeError("empty union")
    return parts[0] if len(parts) == 1 else Manifold.batch_boolean(parts, OpType.Add)


def sub(base, cuts):
    out = base - add(cuts)
    if "NoError" not in str(out.status()):
        raise RuntimeError(f"boolean {out.status()}")
    return out


def sphere(r, c, n=48):
    return Manifold.sphere(float(r), n).translate([float(c[0]), float(c[1]), float(c[2])])


def cyl_z(r, z0, z1, n=48):
    return Manifold.cylinder(z1 - z0, r, r, n).translate([0, 0, z0])


def box(cx, cy, cz, sx, sy, sz):
    return Manifold.cube([sx, sy, sz], True).translate([cx, cy, cz])


def tube(points, radii, n=24):
    bits = []
    for p0, p1, r0, r1 in zip(points, points[1:], radii, radii[1:]):
        bits.append(Manifold.batch_hull([sphere(r0, p0, n), sphere(max(r1, 0.7), p1, n)]))
    return add(bits)


def sector(r0, r1, center, half, z0, z1, n=32):
    a0, a1 = center - half, center + half
    outer = [[r1 * math.cos(a), r1 * math.sin(a)] for a in np.linspace(a0, a1, n)]
    inner = [[r0 * math.cos(a), r0 * math.sin(a)] for a in np.linspace(a1, a0, n)]
    return Manifold.extrude(CrossSection([outer + inner]), z1 - z0).translate([0, 0, z0])


def to_tm(man: Manifold) -> trimesh.Trimesh:
    if man.is_empty() or "NoError" not in str(man.status()):
        raise RuntimeError(f"bad manifold {man.status()} empty={man.is_empty()}")
    raw = man.to_mesh()
    tm = trimesh.Trimesh(
        vertices=np.array(raw.vert_properties)[:, :3],
        faces=np.array(raw.tri_verts),
        process=False,
    )
    tm.merge_vertices()
    tm.update_faces(tm.unique_faces())
    tm.remove_unreferenced_vertices()
    tm.fix_normals()
    if tm.volume < 0:
        tm.invert()
    return tm


def build_head() -> Manifold:
    head = Manifold.sphere(HEAD_R, 72).translate([0, 0, Z_CENTER])
    head = head.trim_by_plane([0, 0, 1], Z_BASE)

    ears = []
    for sign in (-1.0, 1.0):
        base_c = np.array([sign * 14.5, -1.0, Z_CENTER + HEAD_R - 7.5])
        tip_c = np.array([sign * 20.0, -0.6, Z_CENTER + HEAD_R + 16.0])
        mid_c = (base_c + tip_c) / 2 + np.array([sign * 1.0, -0.4, 1.0])
        ear = Manifold.batch_hull(
            [sphere(8.4, base_c, 32), sphere(4.6, mid_c, 28), sphere(1.55, tip_c, 18)]
        )
        pivot = base_c.copy()
        ear = ear.translate((-pivot).tolist()).rotate([0, -sign * 12.0, 0]).translate(pivot.tolist())
        ears.append(ear)

    shell = add([head, *ears])

    # ATOM S3R。ポケットは 25.4 mm、奥行きは本体 12.9 mm + 0.5 mm。
    pz = Z_CENTER + 6.5
    front = 27.2
    depth = 13.4
    pocket = box(0, front - depth / 2, pz, 25.4, depth, 25.4)
    window = box(0, front + 1.5, pz, 20.0, 12.0, 20.0)
    # 後ろから入れる道。天井は半円。
    tunnel = box(0, -8.0, pz, 26.0, 56.0, 25.4)
    arch = cyl_z(13.0, -36.0, 20.0, 36).rotate([0, 90, 0]).translate([0, -8.0, pz + 12.7])
    cable = box(0, -10.0, pz - 12.7 - 5.0, 16.0, 34.0, 14.0)

    cavity = box(
        0,
        (BODY_Y0 + BODY_Y1) / 2,
        FLOOR + BODY_Z / 2,
        BODY_X,
        BODY_Y1 - BODY_Y0,
        BODY_Z,
    )
    tabs = box(
        0,
        (TAB_Y0 + TAB_Y1) / 2,
        FLOOR + (TAB_Z0 + TAB_Z1) / 2,
        BODY_X,
        TAB_Y1 - TAB_Y0,
        TAB_Z1 - TAB_Z0,
    )
    collar = cyl_z(COLLAR_R, Z_BASE - 0.4, FLOOR + 0.3, 40)
    mouth = box(0, -30.0, FLOOR + 1.3 + (BODY_Z + 1.5) / 2, BODY_X + 0.6, 36.0, BODY_Z + 1.5)
    wire = box(0, -8.0, FLOOR + BODY_Z - 2.0, 8.0, 14.0, 12.0)

    # 尻尾が底を横切る扇形。サーボの胴（半径およそ 21 mm）より外から。
    fan = sector(21.0, 52.0, TAIL_AZ, math.radians(52.0), Z_BASE - 0.4, Z_BASE + 30.0)

    holes = []
    for sign in (-1.0, 1.0):
        for zoff, yoff, tilt in ((6.2, 4.0, 0.16), (0.0, 7.5, 0.0), (-6.4, 4.0, -0.15)):
            d = np.array([sign, 0.22, tilt], dtype=float)
            d /= np.linalg.norm(d)
            outer = np.array([sign * 38.0, 12.0 + yoff, pz + zoff])
            holes.append(tube([outer - d * 14.0, outer + d * 1.5], [1.35, 1.35], n=14))

    return sub(shell, [pocket, window, tunnel, arch, cable, cavity, tabs, collar, mouth, wire, fan, *holes])


def tentacle(angle_deg, length, curl):
    a = math.radians(angle_deg)
    dx, dy = math.sin(a), math.cos(a)
    sx, sy = math.cos(a), -math.sin(a)
    pts, radii = [], []
    for i in range(6):
        t = i / 5
        dist = 11.0 + length * t
        # 頭の底（Z_BASE）より下を通す。根元の上端が底に触れない。
        z = 3.0 + (HUB_TOP - 11.0 - 3.0) * ((1 - t) ** 0.85)
        side = curl * 6.0 * t * t
        pts.append((dx * dist + sx * side, dy * dist + sy * side, z))
        radii.append(5.4 + (4.0 - 5.4) * t)
    return tube(pts, radii, n=20)


def build_moving() -> Manifold:
    hub = add(
        [
            sphere(15.5, [0, 0, HUB_TOP - 9.0], 48),
            cyl_z(14.2, HUB_TOP - 14.0, HUB_TOP + 1.0, 56),
        ]
    ).trim_by_plane([0, 0, -1], -HUB_TOP)

    legs = [
        tentacle(a, L, c)
        for a, L, c in (
            (16, 28, 0.65),
            (55, 32, -0.5),
            (95, 34, 0.4),
            (138, 26, -0.3),
            (-16, 28, -0.65),
            (-55, 32, 0.5),
            (-95, 34, -0.4),
            (-138, 26, 0.3),
        )
    ]

    d = az_dir()
    # ハブから 50° 前後で立ち上がり、頭の外を通って浅く巻く。
    stations = [12.0, 22.0, 34.0, 44.0, 50.0, 50.0, 44.0]
    zs = [26.0, 34.0, 46.0, 64.0, 86.0, 106.0, 122.0]
    radii = [5.2, 5.0, 4.8, 4.5, 4.0, 3.5, 2.9]
    pts = [(d[0] * s, d[1] * s, z) for s, z in zip(stations, zs)]
    tail = tube(pts, radii, n=22)

    moving = add([hub, *legs, tail]).trim_by_plane([0, 0, 1], 0.0)
    # 溝を尻尾の方位へ回す。box は軸平行なので、回転してから引く。
    slot_along = box(0, 0, HUB_TOP - 1.55, 6.8, 26.0, 3.5).rotate(
        [0, 0, math.degrees(TAIL_AZ) - 90.0]
    )
    slot_cross = box(0, 0, HUB_TOP - 1.55, 22.0, 6.8, 3.5).rotate(
        [0, 0, math.degrees(TAIL_AZ) - 90.0]
    )
    screw = cyl_z(1.7, HUB_TOP - 16.0, HUB_TOP + 0.4, 20)
    driver = cyl_z(4.8, 6.0, HUB_TOP - 5.0, 24)
    return sub(moving, [slot_along, slot_cross, screw, driver])


def build_whisker() -> Manifold:
    shaft = tube([(0, 0, 1.05), (26, 0, 1.05)], [0.95, 0.7], n=16)
    root = tube([(-0.4, 0, 1.2), (6.5, 0, 1.2)], [1.2, 1.2], n=16)
    return add([shaft, root]).trim_by_plane([0, 0, 1], 0.0)


def servo_probe() -> Manifold:
    body = box(0, (BODY_Y0 + 0.6 + BODY_Y1 - 0.5) / 2, FLOOR + 11.4, 12.2, (BODY_Y1 - 0.5) - (BODY_Y0 + 0.6), 22.4)
    tabs = box(0, (TAB_Y0 + 0.8 + TAB_Y1 - 0.6) / 2, FLOOR + 2.6, 12.4, (TAB_Y1 - 0.6) - (TAB_Y0 + 0.8), 2.6)
    return add([body, tabs])


def atom_probe() -> Manifold:
    pz = Z_CENTER + 6.5
    front = 27.2
    # ガラスが窓側、奥行き 12.9。ポケット前端は front。
    return box(0, front - 0.4 - 12.9 / 2, pz, 24.0, 12.9, 24.0)


def yaw_mesh(tm, deg):
    spun = tm.copy()
    spun.apply_transform(trimesh.transformations.rotation_matrix(math.radians(deg), [0, 0, 1]))
    return spun


def overlap(a: Manifold, b: Manifold) -> float:
    inter = a ^ b
    return 0.0 if inter.is_empty() else float(inter.volume())


def render(head_tm, move_tm, path):
    size = 480
    tiles = []
    for view in ("front", "side", "back", "top"):
        img = Image.new("RGB", (size, size), (235, 235, 235))
        draw = ImageDraw.Draw(img)
        allv = np.vstack([head_tm.vertices, move_tm.vertices])

        def xy_of(v):
            if view == "front":
                return np.stack([v[:, 0], v[:, 2]], 1)
            if view == "side":
                return np.stack([v[:, 1], v[:, 2]], 1)
            if view == "back":
                return np.stack([-v[:, 0], v[:, 2]], 1)
            return np.stack([v[:, 0], -v[:, 1]], 1)

        pts = xy_of(allv)
        lo, hi = pts.min(0), pts.max(0)
        scale = (size - 30) / max(hi - lo)
        mid = (lo + hi) / 2

        def conv(p):
            q = (p - mid) * scale
            return (size / 2 + q[0], size / 2 - q[1])

        for mesh, col in ((head_tm, (32, 32, 36)), (move_tm, (70, 70, 78))):
            step = max(1, len(mesh.faces) // 6000)
            order = mesh.triangles_center[:, 2]
            for i in np.argsort(order)[::step]:
                poly = [tuple(conv(p)) for p in xy_of(mesh.vertices[mesh.faces[i]])]
                draw.polygon(poly, outline=col)
        tiles.append(img)
    sheet = Image.new("RGB", (size * 2, size * 2), (235, 235, 235))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * size, (i // 2) * size))
    sheet.save(path)


def main():
    set_circular_segments(48)
    OUT.mkdir(parents=True, exist_ok=True)
    print("build")
    head_m = build_head()
    move_m = build_moving()
    whisk_m = build_whisker()
    head = to_tm(head_m)
    moving = to_tm(move_m)
    whisker = to_tm(whisk_m)

    for name, mesh in (("頭", head), ("動", moving), ("ひげ", whisker)):
        ext = mesh.extents
        parts = mesh.split(only_watertight=False)
        print(
            f"{name} {ext.round(1).tolist()} watertight={mesh.is_watertight} "
            f"vol={mesh.volume:.0f} faces={len(mesh.faces)} parts={len(parts)}"
        )
        if (not mesh.is_watertight) or ext.max() > 175 or mesh.volume < 20:
            raise SystemExit(f"bad {name}")
    if len(moving.split(only_watertight=True)) != 1:
        raise SystemExit("tail is not joined to the tentacles")

    servo = servo_probe()
    atom = atom_probe()
    print(f"servo in plastic {overlap(servo, head_m):.2f}")
    print(f"atom in plastic {overlap(atom, head_m):.2f}")
    if overlap(servo, head_m) > 5 or overlap(atom, head_m) > 5:
        raise SystemExit("servo or ATOM does not fit")

    glass = box(0, 29.5, Z_CENTER + 6.5, 16.0, 4.0, 16.0)
    print(f"window blocked {overlap(glass, head_m):.2f}")
    if overlap(glass, head_m) > 2:
        raise SystemExit("screen is not pressable")

    from scipy.spatial import cKDTree

    tree = cKDTree(head.vertices)
    for deg in (0, -28, 28, -32, 32):
        spun = yaw_mesh(moving, deg)
        inter = trimesh.boolean.intersection([head, spun], engine="manifold")
        vol = 0.0 if inter is None else float(inter.volume)
        dist, _ = tree.query(spun.vertices[::2])
        print(f"yaw {deg:+4.0f} inter {vol:.2f} gap {dist.min():.2f}")
        if vol > 2.0 or dist.min() < 1.0:
            raise SystemExit(f"hit at {deg}")

    for deg in (0, -32, 32):
        spun = yaw_mesh(moving, deg)
        sm = Manifold.of_mesh(spun.vertices, spun.faces) if hasattr(Manifold, "of_mesh") else None
        if sm is None:
            raw = trimesh.Trimesh(vertices=spun.vertices, faces=spun.faces, process=False)
            # rebuild via to_mesh inverse: use trimesh boolean volume vs servo already covered if servo is inside head void
            servo_tm = to_tm(servo)
            hit = trimesh.boolean.intersection([servo_tm, spun], engine="manifold")
            hv = 0.0 if hit is None else float(hit.volume)
            print(f"servo vs moving {deg:+} {hv:.2f}")
            if hv > 2:
                raise SystemExit("tail hits the servo")

    head.apply_translation([0, 0, -Z_BASE])
    for mesh, fn in ((head, "頭.stl"), (moving, "タコ足と尻尾.stl"), (whisker, "ひげ.stl")):
        mesh.export(OUT / fn)
        print("wrote", fn, mesh.extents.round(1).tolist())

    preview_move = moving.copy()
    preview_move.apply_translation([0, 0, -Z_BASE])
    render(head, preview_move, Path("/tmp/neko_sheet.png"))
    print("preview /tmp/neko_sheet.png")
    print("Z_CENTER", round(Z_CENTER, 2), "base r", round(HEAD_R * math.sin(CUT), 2))


if __name__ == "__main__":
    main()
