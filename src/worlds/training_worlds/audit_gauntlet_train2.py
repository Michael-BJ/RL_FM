#!/usr/bin/env python3
# audit_gauntlet_train2.py — v1.0 (2026-07-08)
"""
Audit laik-lewat + multimodalitas untuk world GAUNTLET_TRAIN2 (tanpa ROS).
Ambang IDENTIK audit_gauntlet_train.py / audit_test01.py:

  1. PASSABILITY  : jalur start->goal dengan clearance >= 0.65 m (planner
                    menerima) DAN >= 0.80 m (bebas penalti).
  2. PINCH AUDIT  : pasangan antar-event >= 1.30 m; pasangan yang melibatkan
                    tiang pengisi >= 2.00 m (tidak ada pengisi di v1 dunia
                    ini, lihat generate_wide_worlds.py, jadi bagian ini
                    selalu lolos trivial -- dipertahankan untuk konsistensi
                    kalau pengisi ditambah nanti).
  3. MULTIMODAL   : tiap event ambigu >= 2 kelas homotopi pada jendela
                    lokal; G3/G6 sengaja 1 (threading, gaya R6/E6).

Jalankan:  python3 audit_gauntlet_train2.py
Output  :  laporan konsol + gauntlet_train2_layout.png (denah + jalur A*).
"""
import itertools
import os
import sys

import numpy as np
from scipy import ndimage

# 2026-07-08: file ini dipindah ke training_worlds/ (satu level lebih dalam
# dari worlds/) -- ".." tambahan supaya path relatif tetap benar.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "fm_planner", "fm_planner"))
sys.path.insert(0, os.path.join(_HERE, ".."))   # generate_wide_worlds.py tetap di worlds/
from homotopy_frontend import HomotopyFrontEnd  # noqa: E402

from generate_wide_worlds import GAUNTLET_TRAIN2  # noqa: E402

RES        = 0.05
X_RANGE    = (-3.0, 34.0)
Y_RANGE    = (-7.0, 7.0)
ARENA_X    = (-2.0, 33.0)  # = arena expert (arena_x_max:=33)
ARENA_Y    = (-6.0, 6.0)
START      = np.array([0.0, 0.0])
GOAL       = np.array([30.0, 0.0])
NEED_HARD  = 0.65          # inflasi 0.3 + edt terima 0.35
NEED_FREE  = 0.80          # inflasi 0.3 + safe_dis 0.5 (bebas penalti)
PAIR_MIN   = 1.30          # 2 x NEED_HARD (batas absolut)
GAP_TARGET = 2.00          # celah fungsional bebas penalti
A_BOUNDS   = (-1.5, 32.5, -5.5, 5.5)   # bounds A* frontend (arena - inset 0.5)

# Baris/event GAUNTLET_TRAIN2 (indeks mengikuti urutan list GAUNTLET_TRAIN2):
ROW_OF = {0: "G1", 1: "G1", 2: "G1",
          3: "G2", 4: "G2",
          5: "G3", 6: "G3", 7: "G3", 8: "G3",
          9: "G4",
          10: "G5", 11: "G5", 12: "G5",
          13: "G6", 14: "G6", 15: "G6", 16: "G6"}

# (nama, start, goal, min kelas homotopi). G3/G5/G6 sengaja 1: threading --
# G3 = gerbang 2-tiang sederhana, G5 = weave 3-segmen (LEBIH SULIT: A*
# empiris hanya menemukan 1 kelas -- swing penuh di luar seluruh klaster
# jauh lebih mahal daripada S-weave, jadi bukan kelas kompetitif kedua;
# diverifikasi, bukan diasumsikan, lihat CHANGELOG audit ini), G6 = slalom
# 4-tiang. Ketiganya beda TIPE threading, bukan duplikat.
EVENTS = [("G1 silinder gemuk 2.2m", (1.0, 0.0),  (6.0, 0.0),  2),
          ("G2 pintu 2.0m (+1.9)",   (5.5, 0.0),  (10.5, 0.0), 2),
          ("G3 gerbang 2 tiang",     (9.5, 0.0),  (15.0, 0.0), 1),
          ("G4 dinding tanpa pintu", (14.0, 0.0), (19.5, 0.0), 2),
          ("G5 zig-zag 3-segmen (threading)", (18.5, 0.0), (26.5, 0.0), 1),
          ("G6 slalom (threading)",  (26.5, 0.0), (29.5, 0.0), 1)]


def build_edt():
    xs = np.arange(X_RANGE[0] + RES / 2, X_RANGE[1], RES)
    ys = np.arange(Y_RANGE[0] + RES / 2, Y_RANGE[1], RES)
    XX, YY = np.meshgrid(xs, ys)               # (rows=y, cols=x)
    occ = np.zeros(XX.shape, bool)
    for ob in GAUNTLET_TRAIN2:
        if ob["type"] == "box":
            occ |= ((np.abs(XX - ob["x"]) <= ob["sx"] / 2)
                    & (np.abs(YY - ob["y"]) <= ob["sy"] / 2))
        else:
            occ |= (XX - ob["x"]) ** 2 + (YY - ob["y"]) ** 2 <= ob["r"] ** 2
    # Dinding virtual arena — identik esdf_ros2 arena_bounds (expert node)
    occ |= ((XX < ARENA_X[0]) | (XX > ARENA_X[1])
            | (YY < ARENA_Y[0]) | (YY > ARENA_Y[1]))
    return ndimage.distance_transform_edt(~occ) * RES, xs, ys


class _Origin:
    x, y = X_RANGE[0] + RES / 2, Y_RANGE[0] + RES / 2


class _MockESDF:
    map_resolution = RES
    map_origin = _Origin()

    def __init__(self, edt):
        self.esdf_map = edt


def surface_gap(a, b):
    def rect_dist(ob, px, py):
        dx = max(abs(px - ob["x"]) - ob["sx"] / 2, 0.0)
        dy = max(abs(py - ob["y"]) - ob["sy"] / 2, 0.0)
        return np.hypot(dx, dy)

    if a["type"] == "cyl" and b["type"] == "cyl":
        return np.hypot(a["x"] - b["x"], a["y"] - b["y"]) - a["r"] - b["r"]
    if a["type"] == "box" and b["type"] == "box":
        dx = max(abs(a["x"] - b["x"]) - (a["sx"] + b["sx"]) / 2, 0.0)
        dy = max(abs(a["y"] - b["y"]) - (a["sy"] + b["sy"]) / 2, 0.0)
        return np.hypot(dx, dy)
    box, cyl = (a, b) if a["type"] == "box" else (b, a)
    return rect_dist(box, cyl["x"], cyl["y"]) - cyl["r"]


def arena_wall_gap(ob):
    """Jarak permukaan obstacle ke dinding virtual arena terdekat."""
    if ob["type"] == "box":
        hx, hy = ob["sx"] / 2, ob["sy"] / 2
        r = 0.0
    else:
        hx = hy = 0.0
        r = ob["r"]
    return min(ob["x"] - hx - ARENA_X[0], ARENA_X[1] - ob["x"] - hx,
               ob["y"] - hy - ARENA_Y[0], ARENA_Y[1] - ob["y"] - hy) - r


def min_edt_along(edt, xs, ys, path_xy):
    ii = np.clip(np.round((path_xy[:, 1] - (Y_RANGE[0] + RES / 2)) / RES)
                 .astype(int), 0, len(ys) - 1)
    jj = np.clip(np.round((path_xy[:, 0] - (X_RANGE[0] + RES / 2)) / RES)
                 .astype(int), 0, len(xs) - 1)
    return float(edt[ii, jj].min())


def main():
    edt, xs, ys = build_edt()
    ok = True

    # ── 2. pinch audit antar-event ──────────────────────────────────────────
    print(f"── Pinch audit (antar-event < {PAIR_MIN:.2f} m ATAU melibatkan "
          f"tiang pengisi < {GAP_TARGET:.2f} m = FAIL) ──")
    worst = (np.inf, None)
    for i, j in itertools.combinations(range(len(GAUNTLET_TRAIN2)), 2):
        if ROW_OF[i] == ROW_OF[j] and ROW_OF[i] != "F":
            continue
        g = surface_gap(GAUNTLET_TRAIN2[i], GAUNTLET_TRAIN2[j])
        if g < worst[0]:
            worst = (g, (i, j))
        involves_filler = ROW_OF[i] == "F" or ROW_OF[j] == "F"
        thresh = GAP_TARGET if involves_filler else PAIR_MIN
        if g < thresh:
            ok = False
            print(f"  FAIL: obstacle #{i}({ROW_OF[i]}) vs #{j}({ROW_OF[j]}) "
                  f"gap {g:.2f} m (ambang {thresh:.2f} m)")
    print(f"  tersempit antar-event: {worst[0]:.2f} m "
          f"(#{worst[1][0]} vs #{worst[1][1]}) -> "
          f"{'OK' if worst[0] >= PAIR_MIN else 'FAIL'}")
    # pengisi -> dinding virtual arena (tidak ada pengisi di v1, trivial OK)
    for i, ob in enumerate(GAUNTLET_TRAIN2):
        if ROW_OF[i] != "F":
            continue
        g = arena_wall_gap(ob)
        flag = "OK " if g >= GAP_TARGET else "FAIL"
        if g < GAP_TARGET:
            ok = False
        print(f"  {flag} pengisi #{i} -> dinding arena: {g:.2f} m "
              f"(ambang {GAP_TARGET:.2f} m)")

    # ── 1. passability start->goal ──────────────────────────────────────────
    esdf = _MockESDF(edt)
    for need, label in [(NEED_HARD, "WAJIB (0.65 m: planner menerima)"),
                        (NEED_FREE, "WAJIB (0.80 m: bebas penalti)")]:
        fe = HomotopyFrontEnd(hard_clearance=need, max_classes=3,
                              bounds=A_BOUNDS)
        paths = fe.find_paths(esdf, START, GOAL)
        tag = f"{len(paths)} jalur" if paths else "TIDAK ADA JALUR"
        print(f"── Passability @ clearance >= {need:.2f} m [{label}]: {tag}")
        for p, sig in paths:
            print(f"    sig={sig}  min-EDT sepanjang jalur "
                  f"{min_edt_along(edt, xs, ys, p):.2f} m")
        if not paths:
            ok = False

    # ── 3. multimodalitas per event ─────────────────────────────────────────
    print("── Kelas homotopi per event (jendela lokal, frontend default) ──")
    fe = HomotopyFrontEnd(bounds=A_BOUNDS)
    all_paths = []
    for name, s, g, need_k in EVENTS:
        paths = fe.find_paths(esdf, np.array(s), np.array(g))
        sigs = [sig for _, sig in paths]
        flag = "OK " if len(paths) >= need_k else "!!  (kurang)"
        if len(paths) < need_k:
            ok = False
        print(f"  {flag} {name:24s} {len(paths)} kelas "
              f"(min {need_k})  sigs={sigs}")
        all_paths.extend(paths)

    # ── denah PNG ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        fig, ax = plt.subplots(figsize=(16, 6))
        for ob in GAUNTLET_TRAIN2:
            if ob["type"] == "box":
                ax.add_patch(Rectangle((ob["x"] - ob["sx"] / 2,
                                        ob["y"] - ob["sy"] / 2),
                                       ob["sx"], ob["sy"],
                                       color="#b03030" if ob["sy"] >= 1.2
                                       else "#5570a8"))
            else:
                ax.add_patch(Circle((ob["x"], ob["y"]), ob["r"],
                                    color="#8a5a2b"))
        for p, sig in all_paths:
            ax.plot(p[:, 0], p[:, 1], lw=1.2, alpha=0.7)
        ax.axhline(6.0, color="k", ls="--", lw=0.8)
        ax.axhline(-6.0, color="k", ls="--", lw=0.8)
        ax.plot(*START, "g^", ms=12, label="start (0,0)")
        ax.plot(*GOAL, "r*", ms=16, label="goal (30,0)")
        for name, s, g, _k in EVENTS:
            ax.annotate(name.split()[0], (s[0] + (g[0] - s[0]) / 2, 5.3),
                        ha="center", fontsize=9)
        ax.set_xlim(-2, 33); ax.set_ylim(-6.5, 6.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title("gauntlet_train2 — koridor 0..30, goal_x:=30, "
                     "arena_x_max:=33 (garis = jalur A* per event)")
        ax.legend(loc="lower right")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gauntlet_train2_layout.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"[png] {out}")
    except ImportError:
        print("[png] matplotlib tidak ada — denah dilewati")

    print("\n=== AUDIT", "LULUS ===" if ok else "GAGAL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
