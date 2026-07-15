#!/usr/bin/env python3
"""decisive_eval.py — recap of the decisive NN vs FM experiment (go/no-go, 2026-07-10).

Reads ~/monitor_logs/<label>_<world>_<timestamp>/summary.txt produced by
flight_monitor.py and tabulates per (model, world). Label convention
(see command.txt "EKSPERIMEN PENENTU" block): use a DASH, not an underscore —
e.g. `nnD-t1`, `fmD-t3` — so `<label>_<world>_<ts>` can be split on the
first underscore. Model = the part of the label before '-t'.

Run directly (no build needed):
    python3 ~/neo_planner_ws/src/fm_planner/fm_planner/decisive_eval.py
Optional: --prefix nnD,fmD (default) to filter other labels.
"""
import argparse
import glob
import os
import re
from collections import defaultdict


def parse_summary(path):
    txt = open(path, errors="replace").read()
    out = {"laps": 0, "escape": 0, "min_clear": None,
           "time_s": None, "path_m": None, "center": 0, "explosion_hint": False}
    m = re.search(r"Laps complete\s*:\s*(\d+)", txt)
    if m:
        out["laps"] = int(m.group(1))
    m = re.search(r"Escape\s*:\s*(\d+)", txt)
    if m:
        out["escape"] = int(m.group(1))
    m = re.search(r"Min clearance\s*:\s*([\d.]+)\s*m", txt)
    if m:
        out["min_clear"] = float(m.group(1))
    m = re.search(r"lap\s+1 .*?:\s*([\d.]+)s\s+path=\s*([\d.]+)m", txt)
    if m:
        out["time_s"], out["path_m"] = float(m.group(1)), float(m.group(2))
    out["center"] = sum(int(n) for n in re.findall(r"(\d+)x CENTER", txt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.path.expanduser("~/monitor_logs"))
    ap.add_argument("--prefix", default="nnD,fmD",
                    help="model prefixes, comma-separated (the part of the label before -t)")
    args = ap.parse_args()
    prefixes = tuple(p.strip() for p in args.prefix.split(","))

    groups = defaultdict(list)   # (model, world) -> [summary dict]
    for d in sorted(glob.glob(os.path.join(args.logdir, "*_*"))):
        base = os.path.basename(d)
        label = base.split("_", 1)[0]
        model = label.split("-t")[0]
        if model not in prefixes:
            continue
        m = re.match(r"[^_]+_(.+)_\d{8}_\d{6}$", base)
        if not m:
            continue
        sp = os.path.join(d, "summary.txt")
        if os.path.isfile(sp):
            groups[(model, m.group(1))].append(parse_summary(sp))

    if not groups:
        print(f"No folders labeled {prefixes} found in {args.logdir}.")
        return

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else float("nan")

    hdr = (f"{'model':6} {'world':14} {'n':>2} {'goal':>4} {'t_med(s)':>9} "
           f"{'path(m)':>9} {'minclr':>7} {'clr<0.2':>7} {'CENTER':>6}")
    print(hdr)
    print("-" * len(hdr))
    for (model, world), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        goal = sum(1 for r in rs if r["laps"] >= 1)
        graze = sum(1 for r in rs
                    if r["min_clear"] is not None and r["min_clear"] < 0.2)
        print(f"{model:6} {world:14} {len(rs):>2} {goal:>2}/{len(rs):<2}"
              f" {med([r['time_s'] for r in rs]):>9.1f}"
              f" {med([r['path_m'] for r in rs]):>9.1f}"
              f" {med([r['min_clear'] for r in rs]):>7.2f}"
              f" {graze:>7} {sum(r['center'] for r in rs):>6}")
    print()
    print("Go/no-go criteria (set 2026-07-10, DO NOT change after seeing results):")
    print("  wall_gauntlet = discriminator. FM is 'go' if the goal-rate and/or")
    print("  CENTER/collision (clr<0.2) rate is consistently better than NN.")
    print("  If FM is NOT better than NN on wall_gauntlet despite a bimodal GATE")
    print("  >=30% -> the core claim fails -> STOP the FM track (negative result).")


if __name__ == "__main__":
    main()
