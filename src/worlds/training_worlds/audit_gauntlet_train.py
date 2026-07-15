#!/usr/bin/env python3
# audit_gauntlet_train.py — v1.0 (2026-07-07)
"""
Audit laik-lewat + multimodalitas untuk world gauntlet_train (tanpa ROS).

Membuktikan 3 hal:
  1. PASSABILITY  : ada jalur start->goal yang SELURUH titiknya berjarak
                    >= 0.65 m dari permukaan obstacle (= inflasi ESDF 0.3 m
                    + syarat terima planner edt>=0.35). Diuji dengan A* dari
                    homotopy_frontend dengan hard_clearance=0.65.
                    Juga dilaporkan level 0.80 m (= koridor bebas penalti,
                    inflasi 0.3 + safe_dis 0.5).
  2. PINCH AUDIT  : semua pasangan obstacle yang bukan satu baris punya jarak
                    permukaan-ke-permukaan >= 1.30 m (2 x 0.65) — tidak ada
                    chokepoint tersembunyi (termasuk pinch diagonal).
  3. MULTIMODAL   : tiap event (R1..R6) menghasilkan >= 2 kelas homotopi pada
                    jendela lokal (persis cara frontend dipakai saat replan).

Jalankan:  python3 audit_gauntlet_train.py
Output  :  laporan konsol + gauntlet_train_layout.png (denah + jalur A*).
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

from generate_wide_worlds import GAUNTLET_TRAIN  # noqa: E402

RES        = 0.05          # grid audit lebih halus dari ESDF asli (0.1)
X_RANGE    = (-3.0, 34.0)  # grid melebihi arena: dinding virtual ikut EDT
Y_RANGE    = (-7.0, 7.0)
ARENA_X    = (-2.0, 33.0)  # = arena expert (arena_x_max:=33); di luar =
ARENA_Y    = (-6.0, 6.0)   #   dinding virtual (esdf_ros2 arena_bounds)
START      = np.array([0.0, 0.0])
GOAL       = np.array([30.0, 0.0])
NEED_HARD  = 0.65          # inflasi 0.3 + edt terima 0.35
NEED_FREE  = 0.80          # inflasi 0.3 + safe_dis 0.5 (bebas penalti)
PAIR_MIN   = 1.30          # 2 x NEED_HARD (batas absolut)
GAP_TARGET = 2.00          # rev 2026-07-07: celah fungsional bebas penalti
                           # (celah 1.5 m terbukti membuat drone berputar)
A_BOUNDS   = (-1.5, 32.5, -5.5, 5.5)   # bounds A* frontend (arena - inset 0.5)

# Baris/event: pasangan DALAM baris yang sama boleh < PAIR_MIN karena celahnya
# memang fitur (pintu 1.5/1.6 m dsb) dan sudah diuji lewat A* passability.
ROW_OF = {0: "R1", 1: "R1", 2: "R1", 3: "R2", 4: "R2",
          5: "R3", 6: "R3", 7: "R3", 8: "R4", 9: "R4",
          10: "R5", 11: "R5", 12: "R5", 13: "R5",
          14: "R6", 15: "R6", 16: "R6", 17: "R6",
          18: "F"}

# (nama, start, goal, min kelas homotopi yang diharapkan). R6 sengaja 1:
# slalom dengan garis lurus bebas = scene threading tak-ambigu (by design).
EVENTS = [("R1 kotak 2.0m",       (1.0, 0.0),  (6.0, 0.0),  2),
          ("R2 pintu 2.0m",       (5.0, 0.0),  (10.0, 0.0), 2),
          ("R3 dinding 4.4m",     (9.5, 0.0),  (14.5, 0.0), 2),
          ("R4 zig-zag",          (13.5, 0.0), (21.0, 0.0), 2),
          ("R5 pintu off-center", (20.0, 0.0), (24.7, 0.0), 2),
          ("R6 slalom (threading)", (24.5, 0.0), (29.3, 0.0), 1)]


def build_edt():
    xs = np.arange(X_RANGE[0] + RES / 2, X_RANGE[1], RES)
    ys = np.arange(Y_RANGE[0] + RES / 2, Y_RANGE[1], RES)
    XX, YY = np.meshgrid(xs, ys)               # (rows=y, cols=x)
    occ = np.zeros(XX.shape, bool)
    for ob in GAUNTLET_TRAIN:
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
    # frontend memetakan sel (r,c) ke world origin + i*res -> origin harus
    # PUSAT sel pertama grid audit (X_RANGE[0] + RES/2), bukan tepinya,
    # supaya jarak EDT tidak bergeser setengah sel.
    x, y = X_RANGE[0] + RES / 2, Y_RANGE[0] + RES / 2


class _MockESDF:
    map_resolution = RES
    map_origin = _Origin()

    def __init__(self, edt):
        self.esdf_map = edt


def surface_gap(a, b):
    """Jarak permukaan-ke-permukaan dua obstacle (2D, konservatif & eksak
    untuk box-box axis-aligned, box-cyl, cyl-cyl)."""
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

    # ── 2. pinch audit antar-baris ──────────────────────────────────────────
    # rev 2026-07-07 (post-mortem explosion #3): pasangan yang MELIBATKAN
    # tiang pengisi ("F") diperlakukan sama seperti antar-baris (tidak boleh
    # membentuk chokepoint tak-disengaja dgn baris lain) dan diuji terhadap
    # GAP_TARGET (2.0 m), bukan cuma PAIR_MIN (1.3 m). Insiden nyata: tiang
    # pengisi (9.8,-2.6) hanya 1.64/1.70 m dari dinding R2/R3 — lolos 1.3 m
    # tapi menciptakan koridor efektif sempit (esdf terbang 0.6-0.9 m
    # berulang -> escape ping-pong -> collision). Pasangan antar-baris
    # SELAIN "F" tetap diuji PAIR_MIN saja (celah desain yang disengaja,
    # mis. pintu, boleh di bawah 2.0 m selama >= 1.3 m DAN sudah dibuktikan
    # laik-lewat oleh passability A* per-event di bawah).
    print(f"── Pinch audit (antar-baris < {PAIR_MIN:.2f} m ATAU melibatkan "
          f"tiang pengisi < {GAP_TARGET:.2f} m = FAIL) ──")
    worst = (np.inf, None)
    for i, j in itertools.combinations(range(len(GAUNTLET_TRAIN)), 2):
        if ROW_OF[i] == ROW_OF[j] and ROW_OF[i] != "F":
            continue
        g = surface_gap(GAUNTLET_TRAIN[i], GAUNTLET_TRAIN[j])
        if g < worst[0]:
            worst = (g, (i, j))
        involves_filler = ROW_OF[i] == "F" or ROW_OF[j] == "F"
        thresh = GAP_TARGET if involves_filler else PAIR_MIN
        if g < thresh:
            ok = False
            print(f"  FAIL: obstacle #{i}({ROW_OF[i]}) vs #{j}({ROW_OF[j]}) "
                  f"gap {g:.2f} m (ambang {thresh:.2f} m)")
    print(f"  tersempit antar-baris: {worst[0]:.2f} m "
          f"(#{worst[1][0]} vs #{worst[1][1]}) -> "
          f"{'OK' if worst[0] >= PAIR_MIN else 'FAIL'}")

    # ── 1. passability start->goal ──────────────────────────────────────────
    esdf = _MockESDF(edt)
    for need, label in [(NEED_HARD, "WAJIB (0.65 m: planner menerima)"),
                        (NEED_FREE, "WAJIB (0.80 m: bebas penalti, rev "
                                    "2026-07-07)")]:
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
        for ob in GAUNTLET_TRAIN:
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
        ax.set_title("gauntlet_train — koridor 0..30, goal_x:=30, "
                     "arena_x_max:=33 (garis = jalur A* per event)")
        ax.legend(loc="lower right")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gauntlet_train_layout.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"[png] {out}")
    except ImportError:
        print("[png] matplotlib tidak ada — denah dilewati")

    print("\n=== AUDIT", "LULUS ===" if ok else "GAGAL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
