#!/usr/bin/env python3
"""Printable black-cat / octopus shell for the robokko electronics.

One SG90 yaws a single moving piece (tentacles + tail) under a fixed head.
The head holds the ATOM S3R stacked on the Atomic Battery Base from the
macaron build. Stock hinge is not used.

Run from the repo root: python3 models/印刷部品/黒猫タコ/generate.py
"""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent

# Servo shaft points down. Horn and the moving skirt sit under the body.
SHAFT_X = 0.0
SHAFT_Y = -8.0
SERVO_BOTTOM_Z = 40.0
SHAFT_HOLE_Z = SERVO_BOTTOM_Z - 1.2

# SG90 body, shaft offset toward the front (+Y) from the body center.
SERVO_BODY_X = 12.6
SERVO_BODY_Y = 23.2
SERVO_TAB_Y = 32.6
SERVO_BODY_Z = 23.0
SHAFT_OFFSET_Y = 4.0

HEAD_CENTER = np.array([0.0, 24.0, 106.0])
HEAD_R = np.array([33.0, 30.0, 29.0])


def _align_z(direction):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)


def rod(p0, p1, radius, sections=20):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = np.linalg.norm(p1 - p0)
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    mesh.apply_transform(_align_z(p1 - p0))
    mesh.apply_translation((p0 + p1) / 2.0)
    ball = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    a = ball.copy()
    b = ball.copy()
    a.apply_translation(p0)
    b.apply_translation(p1)
    return trimesh.boolean.union([mesh, a, b], engine="manifold")


def box_at(center, extents):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def ellipsoid(radii, subdivisions=4):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(radii)
    return mesh


def union_all(meshes):
    acc = meshes[0]
    for mesh in meshes[1:]:
        acc = trimesh.boolean.union([acc, mesh], engine="manifold")
    return acc


def difference_all(mesh, cutters):
    for cutter in cutters:
        mesh = trimesh.boolean.difference([mesh, cutter], engine="manifold")
    return mesh


def build_fixed():
    head = ellipsoid(HEAD_R)
    head.apply_translation(HEAD_CENTER)

    ears = []
    for sign in (-1, 1):
        ear = ellipsoid(np.array([9.5, 6.5, 16.0]), subdivisions=3)
        ear.apply_translation([sign * 16.0, -2.0, 22.0] + HEAD_CENTER)
        ears.append(ear)

    # Neck drops from the head onto the servo cage.
    neck = box_at([SHAFT_X, 2.0, 82.0], [16.0, 30.0, 36.0])
    servo_block = box_at(
        [SHAFT_X, SHAFT_Y, SERVO_BOTTOM_Z + 13.0],
        [16.0, SERVO_TAB_Y + 4.0, 26.0],
    )
    outer = union_all([head, neck, servo_block, *ears])

    inner = ellipsoid(HEAD_R - 2.8)
    inner.apply_translation(HEAD_CENTER)

    # Chin opening and a square well big enough for ATOM (24) + battery base (24).
    # The stack is inserted upward, then forward against the screen bezel.
    well = box_at(
        [HEAD_CENTER[0], HEAD_CENTER[1] - 2.0, HEAD_CENTER[2] - 6.0],
        [26.2, 42.0, 40.0],
    )
    chin = box_at([0.0, 6.0, 74.0], [28.0, 34.0, 22.0])

    # Screen window. Active area is ~15 mm; 18.6 mm leaves the glass pressable
    # and a lip on the 24 mm body.
    window = box_at(
        [HEAD_CENTER[0], HEAD_CENTER[1] + HEAD_R[1] - 2.0, HEAD_CENTER[2] + 3.0],
        [18.6, 16.0, 18.6],
    )
    bezel = box_at(
        [HEAD_CENTER[0], HEAD_CENTER[1] + 16.0, HEAD_CENTER[2] + 3.0],
        [30.0, 8.0, 30.0],
    )

    # Battery switch and USB reach the right and lower edges of the stack.
    switch_slot = box_at([16.5, HEAD_CENTER[1] - 4.0, HEAD_CENTER[2] - 4.0], [16.0, 14.0, 10.0])

    # Servo drops in from the rear. Floor keeps the body in; a slot lets the
    # shaft out. The horn, once screwed on, stops the servo sliding back.
    servo_cavity = box_at(
        [SHAFT_X, SHAFT_Y + 2.0, SERVO_BOTTOM_Z + SERVO_BODY_Z / 2.0 + 1.0],
        [SERVO_BODY_X + 0.6, SERVO_TAB_Y + 8.0, SERVO_BODY_Z + 1.0],
    )
    shaft_hole = trimesh.creation.cylinder(radius=4.6, height=12.0, sections=24)
    shaft_hole.apply_translation([SHAFT_X, SHAFT_Y + SHAFT_OFFSET_Y, SERVO_BOTTOM_Z - 2.0])

    shell = difference_all(outer, [inner])
    shell = union_all([shell, bezel])
    shell = difference_all(shell, [well, chin, window, switch_slot, servo_cavity, shaft_hole])

    whiskers = []
    for sign in (-1, 1):
        for dz, dy in ((-7.0, 4.0), (-1.0, 8.0), (5.0, 4.0)):
            start = HEAD_CENTER + np.array([sign * 18.0, 10.0 + dy, dz])
            end = HEAD_CENTER + np.array([sign * 40.0, 16.0 + dy, dz])
            whiskers.append(rod(start, end, 1.25, sections=12))
    shell = difference_all(shell, whiskers)
    return shell


def tentacle_curves():
    """Eight legs around the shaft, leaving the rear open for the tail root."""
    curves = []
    angles = [20, 55, 90, 125, 160, 205, 250, 330]
    for index, angle in enumerate(angles):
        curl = 0.55 if index % 2 == 0 else -0.55
        length = 30.0 if 70 <= angle <= 120 else 36.0
        direction = np.deg2rad(angle)
        radius0 = 16.0
        point = np.array(
            [
                SHAFT_X + np.cos(direction) * radius0,
                SHAFT_Y + SHAFT_OFFSET_Y + np.sin(direction) * radius0,
                29.0,
            ]
        )
        pts = [point.copy()]
        for step in range(7):
            direction += curl * 0.16
            travel = length / 7.0
            point = point + travel * np.array([np.cos(direction), np.sin(direction), 0.0])
            point[2] = 29.0
            pts.append(point.copy())
        curves.append(pts)
    return curves


def build_moving():
    hub = trimesh.creation.cylinder(radius=18.0, height=6.0, sections=40)
    hub.apply_translation([SHAFT_X, SHAFT_Y + SHAFT_OFFSET_Y, 29.5])

    legs = [hub]
    for curve in tentacle_curves():
        radii = np.linspace(5.2, 3.2, len(curve) - 1)
        for p0, p1, radius in zip(curve, curve[1:], radii):
            legs.append(rod(p0, p1, radius, sections=14))
    skirt = union_all(legs)

    # Tail leaves the rear of the hub and curls up beside the head.
    tail_pts = []
    for t in np.linspace(0.0, 1.0, 16):
        # Stay low until the tail is behind the head, then rise.
        rise = max(0.0, (t - 0.28) / 0.72)
        tail_pts.append(
            [
                SHAFT_X + 26.0 * rise,
                SHAFT_Y + SHAFT_OFFSET_Y - 14.0 - 50.0 * t,
                30.0 + 62.0 * (rise ** 1.15),
            ]
        )
    tail_bits = []
    for p0, p1, radius in zip(tail_pts, tail_pts[1:], np.linspace(6.5, 3.2, len(tail_pts) - 1)):
        tail_bits.append(rod(p0, p1, radius, sections=14))
    moving = union_all([skirt, *tail_bits])

    # Pocket for the single-arm horn. Screw head sits in a recess.
    slot = box_at(
        [SHAFT_X, SHAFT_Y + SHAFT_OFFSET_Y - 8.0, 32.2],
        [7.0, 20.0, 3.2],
    )
    screw = trimesh.creation.cylinder(radius=3.4, height=8.0, sections=16)
    screw.apply_translation([SHAFT_X, SHAFT_Y + SHAFT_OFFSET_Y, 33.2])
    pin = trimesh.creation.cylinder(radius=1.15, height=16.0, sections=12)
    pin.apply_transform(_align_z([1.0, 0.0, 0.0]))
    pin.apply_translation([SHAFT_X, SHAFT_Y + SHAFT_OFFSET_Y - 10.0, 31.4])
    moving = difference_all(moving, [slot, screw, pin])
    return moving


def build_whisker():
    shaft = trimesh.creation.cylinder(radius=1.05, height=22.0, sections=12)
    shaft.apply_transform(_align_z([1.0, 0.0, 0.0]))
    collar = trimesh.creation.cylinder(radius=2.2, height=1.6, sections=16)
    collar.apply_transform(_align_z([1.0, 0.0, 0.0]))
    collar.apply_translation([-6.0, 0.0, 0.0])
    return trimesh.boolean.union([shaft, collar], engine="manifold")


def export_mesh(mesh, name):
    mesh.merge_vertices()
    mesh.fix_normals()
    path = OUT / name
    mesh.export(path)
    print(
        f"{name}: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm"
        f"  watertight={mesh.is_watertight}  volume={mesh.volume:.0f}"
    )
    return path


def preview(fixed, moving):
    image = Image.new("RGB", (900, 520), (245, 242, 236))
    draw = ImageDraw.Draw(image)

    def paint(mesh, color, origin, scale, flip_y=True):
        verts = mesh.vertices
        for tri in mesh.faces[:: max(1, len(mesh.faces) // 2500)]:
            pts = []
            for index in tri:
                x, y, z = verts[index]
                px = origin[0] + x * scale
                py = origin[1] + ((-z if flip_y else y) * scale)
                pts.append((px, py))
            draw.polygon(pts, outline=color)

    paint(fixed, (40, 40, 40), (230, 470), 3.1)
    paint(moving, (180, 90, 40), (230, 470), 3.1)
    # top view
    verts_f = fixed.vertices
    verts_m = moving.vertices
    for mesh_verts, color, ox in ((verts_f, (40, 40, 40), 640), (verts_m, (180, 90, 40), 640)):
        pass
    for mesh, color in ((fixed, (40, 40, 40)), (moving, (180, 90, 40))):
        for tri in mesh.faces[:: max(1, len(mesh.faces) // 2000)]:
            pts = []
            for index in tri:
                x, y, _z = mesh.vertices[index]
                pts.append((ox + 180 + x * 2.4, 260 - y * 2.4))
            draw.polygon(pts, outline=color)
    path = OUT / "preview.png"
    image.save(path)
    print("preview", path)


def yaw_clearance(fixed, moving):
    """Vertex gap between the fixed shell and the skirt swung ±28°."""
    tree_points = fixed.vertices[::3]
    from scipy.spatial import cKDTree

    tree = cKDTree(tree_points)
    worst = 999.0
    for degrees in (-28, 28):
        spun = moving.copy()
        spun.apply_translation([-SHAFT_X, -(SHAFT_Y + SHAFT_OFFSET_Y), 0.0])
        rotation = trimesh.transformations.rotation_matrix(
            np.deg2rad(degrees), [0, 0, 1]
        )
        spun.apply_transform(rotation)
        spun.apply_translation([SHAFT_X, SHAFT_Y + SHAFT_OFFSET_Y, 0.0])
        dists, _ = tree.query(spun.vertices[::4], k=1)
        worst = min(worst, float(dists.min()))
        print(f"clearance at {degrees:+d} deg: {dists.min():.2f} mm")
    return worst


def main():
    fixed = build_fixed()
    moving = build_moving()
    whisker = build_whisker()
    export_mesh(fixed, "頭と台.stl")
    export_mesh(moving, "足としっぽ.stl")
    export_mesh(whisker, "ひげ.stl")
    preview(fixed, moving)
    gap = yaw_clearance(fixed, moving)
    if gap < 2.0:
        raise SystemExit(f"moving parts come within {gap:.2f} mm of the head")


if __name__ == "__main__":
    main()
