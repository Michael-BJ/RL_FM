#!/usr/bin/env python3
# audit_maze_train.py — v1.0 (2026-07-21)
"""
Audit laik-lewat + multimodalitas untuk world MAZE_TRAIN (tanpa ROS).
Ambang IDENTIK audit_gauntlet_train.py / audit_gauntlet_train2.py:

  1. PASSABILITY  : jalur start->goal dengan clearance >= 0.65 m (planner
                    menerima) DAN >= 0.80 m (bebas penalti).
  2. PINCH AUDIT  : pasangan antar-event >= 1.30 m (batas absolut). Target
                    fungsional user 1.8 m dilaporkan eksplisit: FAIL bila
                    tersempit antar-event < 1.80 m.
  3. MULTIMODAL   : titik pilihan (C1/C2) >= 2 kelas homotopi; pintu berliku
                    (D1..D5) sengaja 1 kelas (threading lewat satu pintu).

Jalankan:  python3 audit_maze_train.py
Output  :  laporan konsol + maze_train_layout.png (denah + jalur A*).
"""
import itertools
import os
import sys

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "fm_planner", "fm_planner"))
sys.path.insert(0, os.path.join(_HERE, ".."))   # generate_wide_worlds.py di worlds/
from homotopy_frontend import HomotopyFrontEnd  # noqa: E402

from generate_wide_worlds import MAZE_TRAIN  # noqa: E402

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
GAP_TARGET = 1.80          # target fungsional user (jarak minimum antar-obstacle)
A_BOUNDS   = (-1.5, 32.5, -5.5, 5.5)   # bounds A* frontend (arena - inset 0.5)

# Event MAZE_TRAIN (indeks mengikuti urutan list MAZE_TRAIN):
#   D1 pintu +1.6 | D2 pintu -1.6 | C1 bimodal | D3 pintu +1.6 |
#   D4 pintu -1.6 | C2 bimodal | D5 pintu +1.6
ROW_OF = {0: "D1", 1: "D1",
          2: "D2", 3: "D2",
          4: "C1", 5: "C1", 6: "C1",
          7: "D3", 8: "D3",
          9: "D4", 10: "D4",
          11: "C2", 12: "C2", 13: "C2",
          14: "D5", 15: "D5"}

# (nama, start, goal, min kelas homotopi). D1..D5 = pintu tunggal -> 1 kelas
# (threading). C1/C2 = box tengah + 2 tiang -> 2 kelas (kiri/kanan).
EVENTS = [("D1 pintu +1.6 (anyam atas)",  (2.0, 0.0),  (6.0, 0.0),  1),
          ("D2 pintu -1.6 (anyam bawah)", (6.0, 0.0),  (10.0, 0.0), 1),
          ("C1 bimodal box 2.8",          (10.0, 0.0), (14.0, 0.0), 2),
          ("D3 pintu +1.6 (anyam atas)",  (14.0, 0.0), (18.0, 0.0), 1),
          ("D4 pintu -1.6 (anyam bawah)", (18.0, 0.0), (22.0, 0.0), 1),
          ("C2 bimodal box 3.2",          (22.0, 0.0), (26.0, 0.0), 2),
          ("D5 pintu +1.6 (anyam atas)",  (26.0, 0.0), (30.0, 0.0), 1)]


def build_edt():
    xs = np.arange(X_RANGE[0] + RES / 2, X_RANGE[1], RES)
    ys = np.arange(Y_RANGE[0] + RES / 2, Y_RANGE[1], RES)
    XX, YY = np.meshgrid(xs, ys)               # (rows=y, cols=x)
    occ = np.zeros(XX.shape, bool)
    for ob in MAZE_TRAIN:
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


def min_edt_along(edt, xs, ys, path_xy):
    ii = np.clip(np.round((path_xy[:, 1] - (Y_RANGE[0] + RES / 2)) / RES)
                 .astype(int), 0, len(ys) - 1)
    jj = np.clip(np.round((path_xy[:, 0] - (X_RANGE[0] + RES / 2)) / RES)
                 .astype(int), 0, len(xs) - 1)
    return float(edt[ii, jj].min())


def main():
    edt, xs, ys = build_edt()
    ok = True

    # ── 2. pinch audit antar-event (target user 1.8 m) ───────────────────────
    print(f"── Pinch audit (antar-event < {GAP_TARGET:.2f} m = FAIL; "
          f"batas absolut {PAIR_MIN:.2f} m) ──")
    worst = (np.inf, None)
    for i, j in itertools.combinations(range(len(MAZE_TRAIN)), 2):
        if ROW_OF[i] == ROW_OF[j]:      # celah DALAM event = passage (dicek passability)
            continue
        g = surface_gap(MAZE_TRAIN[i], MAZE_TRAIN[j])
        if g < worst[0]:
            worst = (g, (i, j))
        if g < GAP_TARGET:
            ok = False
            print(f"  FAIL: obstacle #{i}({ROW_OF[i]}) vs #{j}({ROW_OF[j]}) "
                  f"gap {g:.2f} m (ambang {GAP_TARGET:.2f} m)")
    print(f"  tersempit antar-event: {worst[0]:.2f} m "
          f"(#{worst[1][0]} vs #{worst[1][1]}) -> "
          f"{'OK (>= 1.8 m)' if worst[0] >= GAP_TARGET else 'FAIL'}")

    # ── passage DALAM event (box<->tiang di C1/C2, dua sisi pintu) ────────────
    print("── Lebar passage dalam event (celah yang harus dilewati) ──")
    worst_pass = (np.inf, None)
    for row in ("C1", "C2"):
        idx = [k for k, v in ROW_OF.items() if v == row]
        for i, j in itertools.combinations(idx, 2):
            g = surface_gap(MAZE_TRAIN[i], MAZE_TRAIN[j])
            if g < worst_pass[0]:
                worst_pass = (g, (row, i, j))
    if worst_pass[1] is not None:
        g = worst_pass[0]
        print(f"  passage tersempit (box<->tiang): {g:.2f} m -> "
              f"{'OK (>= 1.8 m)' if g >= GAP_TARGET else 'FAIL'}")
        if g < GAP_TARGET:
            ok = False

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
        print(f"  {flag} {name:28s} {len(paths)} kelas "
              f"(min {need_k})  sigs={sigs}")
        all_paths.extend(paths)

    # ── denah PNG ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        fig, ax = plt.subplots(figsize=(16, 6))
        for ob in MAZE_TRAIN:
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
        ax.set_title("maze_train — labirin berliku, koridor 0..30, "
                     "goal_x:=30, arena_x_max:=33 (garis = jalur A* per event)")
        ax.legend(loc="lower right")
        out = os.path.join(_HERE, "maze_train_layout.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"[png] {out}")
    except ImportError:
        print("[png] matplotlib tidak ada — denah dilewati")

    print("\n=== AUDIT", "LULUS ===" if ok else "GAGAL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
