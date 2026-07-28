#!/usr/bin/env python3
"""
test_anticollision_guards.py
============================
Offline replay of the 2026-07-27 gauntlet_train2 wall-contact crashes against
the hardened inference logic (see the ANTI-COLLISION HARDENING block in
fm_inference_base.py). No ROS, no Gazebo — the checked methods only touch
`self`, so they are bound to a stub carrying the state RECORDED IN THE LOGS at
the moment of each failure.

Run:  python3 test_anticollision_guards.py
"""

import sys
import types

import numpy as np

import fm_inference_base as fib
from fm_inference_base import (
    BODY_RADIUS_M, POST_CHECK_FLOOR_CENTER, MAX_PLAN_COST,
    SPEED_LIMIT_FLOOR, LOOKAHEAD_HORIZON_S, LOOKAHEAD_MIN_DIST_M,
    LOOKAHEAD_MAX_HORIZON_S, HARD_DIS_CENTER, POST_CHECK_CLEAR_MIN,
    GUARD_CLEAR_CENTER, K_SPEED_MARGIN, FMInferenceBase)


class FakeESDF:
    """ESDF stub. `clear_at(xy) -> body-edge distance`, the frame the real
    get_edt_dis() returns (obstacles pre-inflated by BODY_RADIUS_M)."""

    def __init__(self, clear_fn, grad=(0.0, 0.0)):
        self._fn = clear_fn
        self.grad = grad

    def get_edt_dis(self, pos):
        return float(self._fn(np.asarray(pos, dtype=float)))

    def get_edt_grad(self, pos):
        return list(self.grad)

    def is_ready(self):
        return True

    def is_unobserved(self, pos):
        return False


class FakePlanner:
    def __init__(self, cost, pos_array):
        self.final_cost = cost
        self._pos = np.asarray(pos_array, dtype=float)

    def get_pos_array(self):
        return self._pos


def make_stub(esdf=None, planner=None, drone_xy=(0.0, 0.0),
              max_plan_cost=MAX_PLAN_COST):
    """Minimal object exposing exactly what the checked methods read."""
    s = types.SimpleNamespace()
    s._esdf = esdf
    s._planner = planner
    s._hard_clearance = POST_CHECK_CLEAR_MIN
    s._guard_clear = GUARD_CLEAR_CENTER
    s._speed_margin_k = K_SPEED_MARGIN
    s._max_plan_cost = max_plan_cost
    s._n_cost_reject = 0
    s._n_graze_reject = 0
    s._drone_state = types.SimpleNamespace(
        global_pos=np.array([drone_xy[0], drone_xy[1], 2.0]))
    # bind the real implementations
    s._check_plan_quality = types.MethodType(
        FMInferenceBase._check_plan_quality, s)
    s._validate_planned_traj = types.MethodType(
        FMInferenceBase._validate_planned_traj, s)
    return s


def rejects(fn):
    """True if fn() raises (i.e. the plan is vetoed)."""
    try:
        fn()
        return False
    except Exception:
        return True


FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got={got} want={want}")
    if not ok:
        FAILS.append(name)


# ── 1. Plan-quality gate ─────────────────────────────────────────────────────
def test_cost_gate():
    print("\n1. Plan-quality gate (HOLE 2) — costs taken from the run logs")
    # Healthy percentiles measured over 1434 accepted plans this session.
    for cost in (2.0, 9.0, 12.6, 24.1, 49.0):
        s = make_stub(planner=FakePlanner(cost, [[0, 0]]))
        check(f"healthy cost={cost} accepted",
              rejects(s._check_plan_quality), False)
    # The plans that actually preceded wall contact.
    for cost in (113.1, 141.4, 259.0, 2007.6):
        s = make_stub(planner=FakePlanner(cost, [[0, 0]]))
        check(f"diverged cost={cost} REJECTED",
              rejects(s._check_plan_quality), True)
    # Non-finite solve.
    s = make_stub(planner=FakePlanner(float("inf"), [[0, 0]]))
    check("cost=inf REJECTED", rejects(s._check_plan_quality), True)
    s = make_stub(planner=FakePlanner(float("nan"), [[0, 0]]))
    check("cost=nan REJECTED", rejects(s._check_plan_quality), True)
    # Gate disabled -> old behaviour.
    s = make_stub(planner=FakePlanner(2007.6, [[0, 0]]), max_plan_cost=0.0)
    check("gate disabled accepts 2007.6",
          rejects(s._check_plan_quality), False)


# ── 2. Post-check floor + outward requirement ────────────────────────────────
def test_post_check():
    print("\n2. Post-check (HOLE 1) — run gb state: drone at centre clearance"
          " 0.45 m")
    # A wall at x = 17.0; body-edge clearance falls off linearly westwards.
    # Drone at x=15.8 -> edge 0.15 -> centre 0.45, exactly as logged.
    def wall_field(p):
        return max(0.0, (17.0 - p[0]) - 1.05)

    esdf = FakeESDF(wall_field)
    drone = (15.8, 0.0)
    cur_centre = wall_field(np.array(drone)) + BODY_RADIUS_M
    check("recreated drone clearance == logged 0.45",
          round(cur_centre, 2), 0.45)

    # (a) the killer: a path that stays at ~0.40 m centre, i.e. hugs the wall.
    hug = [[15.8 + 0.05 * i, 0.0] for i in range(20)]
    hug = [p for p in hug if wall_field(np.array(p)) + BODY_RADIUS_M >= 0.40]
    s = make_stub(esdf=esdf, planner=FakePlanner(5.0, hug), drone_xy=drone)
    check("wall-hugging plan REJECTED", rejects(s._validate_planned_traj), True)

    # (b) straight into the wall.
    into = [[15.8 + 0.1 * i, 0.0] for i in range(12)]
    s = make_stub(esdf=esdf, planner=FakePlanner(5.0, into), drone_xy=drone)
    check("plan into the wall REJECTED",
          rejects(s._validate_planned_traj), True)

    # (c) the legitimate case the adaptive rule exists for: back out westwards.
    out = [[15.8 - 0.1 * i, 0.0] for i in range(12)]
    s = make_stub(esdf=esdf, planner=FakePlanner(5.0, out), drone_xy=drone)
    check("outward escape plan ACCEPTED",
          rejects(s._validate_planned_traj), False)

    # (d) with clear space around, a normal plan is unaffected.
    esdf_open = FakeESDF(lambda p: 3.0)
    normal = [[i * 0.2, 0.0] for i in range(20)]
    s = make_stub(esdf=esdf_open, planner=FakePlanner(8.0, normal))
    check("open-space plan ACCEPTED", rejects(s._validate_planned_traj), False)

    # (e) the floor: a plan may not dip to the contact line while the drone
    # still has room above it.
    s = make_stub(esdf=esdf,
                  planner=FakePlanner(5.0, [[15.8, 0.0], [16.35, 0.0]]),
                  drone_xy=drone)
    check("plan dipping below POST_CHECK_FLOOR_CENTER REJECTED",
          rejects(s._validate_planned_traj), True)

    # (f) NO DEADLOCK: a drone already INSIDE the floor must still be able to
    # plan its way out — the floor follows it down instead of rejecting
    # everything (a trajectory always starts at the drone's own position).
    deep_drone = (16.35, 0.0)      # body-edge 0.0 -> centre 0.30, under 0.45
    deep_cur = wall_field(np.array(deep_drone)) + BODY_RADIUS_M
    check("stuck-drone clearance is below the floor",
          deep_cur < POST_CHECK_FLOOR_CENTER, True)
    out_deep = [[16.35 - 0.1 * i, 0.0] for i in range(12)]
    s = make_stub(esdf=esdf, planner=FakePlanner(5.0, out_deep),
                  drone_xy=deep_drone)
    check("stuck drone can still plan OUTWARD",
          rejects(s._validate_planned_traj), False)
    in_deep = [[16.35 + 0.02 * i, 0.0] for i in range(12)]
    s = make_stub(esdf=esdf, planner=FakePlanner(5.0, in_deep),
                  drone_xy=deep_drone)
    check("stuck drone still cannot plan DEEPER",
          rejects(s._validate_planned_traj), True)


# ── 3. Clearance-proportional speed limit ────────────────────────────────────
def v_allow(clear_centre, heading=(1.0, 0.0), v_max=0.9):
    """Call the SHIPPED limiter. A wall sits to the east; `heading` is the
    drone's velocity, so heading=(1,0) closes on it and (0,1) runs parallel."""
    # Uniform field with the obstacle to the +x side: clearance falls off in x,
    # so the gradient (pointing away from the obstacle) is -x.
    esdf = FakeESDF(lambda p: clear_centre - BODY_RADIUS_M - (p[0] - 0.0))
    esdf.grad = (-1.0, 0.0)
    s = make_stub(esdf=esdf)
    s._v_max = v_max
    s._speed_allowance = types.MethodType(
        FMInferenceBase._speed_allowance, s)
    v, _ = s._speed_allowance(np.array([0.0, 0.0]),
                              np.array(heading, dtype=float))
    return v


def test_speed_limit():
    print("\n3. Clearance-proportional speed limit (HOLE 3)")
    # Mid-gap of a 2.0 m corridor: threading must NOT be slowed to a crawl.
    check("mid-gap 2.0 m corridor (clearance 1.00) keeps v_max",
          round(v_allow(1.00), 2), 0.90)
    check("clearance 0.95 keeps v_max", round(v_allow(0.95), 2), 0.90)
    # Closing on a wall: speed decays so hover can actually hold.
    check("clearance 0.85 -> 0.60 m/s", round(v_allow(0.85), 2), 0.60)
    check("clearance 0.75 -> 0.20 m/s", round(v_allow(0.75), 2), 0.20)
    check("clearance 0.70 (trip line) -> floor",
          round(v_allow(0.70), 2), round(SPEED_LIMIT_FLOOR, 2))
    # The logged gb approach: 0.69 m/s at centre clearance 0.45 would have been
    # limited to the floor long before contact.
    check("gb contact state (clearance 0.45) -> floor",
          round(v_allow(0.45), 2), round(SPEED_LIMIT_FLOOR, 2))
    check("floor is non-zero (tight gaps stay threadable)",
          SPEED_LIMIT_FLOOR > 0.0, True)

    print("   closing-rate scaling (wall to the EAST, same 0.75 m clearance):")
    check("heading INTO the wall is throttled",
          round(v_allow(0.75, heading=(1.0, 0.0)), 2), 0.20)
    check("flying PARALLEL to the wall keeps v_max",
          round(v_allow(0.75, heading=(0.0, 1.0)), 2), 0.90)
    check("flying AWAY from the wall keeps v_max",
          round(v_allow(0.75, heading=(-1.0, 0.0)), 2), 0.90)
    # 45 deg: margin needed scales with the closing component (cos45 = 0.707).
    diag = v_allow(0.75, heading=(1.0, 1.0))
    check("45 deg approach is between the two",
          0.20 < round(diag, 2) < 0.90, True)


# ── 4. Look-ahead horizon in metres ──────────────────────────────────────────
def horizon_for(speed):
    if speed <= 0.05:
        return LOOKAHEAD_HORIZON_S
    return float(np.clip(LOOKAHEAD_MIN_DIST_M / speed,
                         LOOKAHEAD_HORIZON_S, LOOKAHEAD_MAX_HORIZON_S))


def test_lookahead():
    print("\n4. Look-ahead horizon covers braking distance")
    for v in (0.69, 0.9, 1.0):
        old_m = LOOKAHEAD_HORIZON_S * v
        new_m = horizon_for(v) * v
        ok = new_m >= LOOKAHEAD_MIN_DIST_M - 1e-6
        print(f"  [{'ok ' if ok else 'FAIL'}] v={v} m/s: "
              f"old scan {old_m:.2f} m -> new {new_m:.2f} m")
        if not ok:
            FAILS.append(f"lookahead v={v}")
    check("slow crawl does not scan past the cap",
          horizon_for(0.2) <= LOOKAHEAD_MAX_HORIZON_S, True)
    check("stationary keeps the original horizon",
          horizon_for(0.0), LOOKAHEAD_HORIZON_S)


# ── 5. No escape anywhere in the hardened paths ──────────────────────────────
def test_no_escape():
    print("\n5. The hardening never calls escape")
    import inspect
    for fn in (FMInferenceBase._check_plan_quality,
               FMInferenceBase._validate_planned_traj,
               FMInferenceBase._speed_allowance):
        src = inspect.getsource(fn)
        check(f"{fn.__name__} escape-free",
              "_enter_escape" in src, False)
    # The limiter must scale the speed only — never steer. Checked
    # behaviourally: the capped velocity stays parallel to the original.
    esdf = FakeESDF(lambda p: 0.75 - BODY_RADIUS_M - p[0], grad=(-1.0, 0.0))
    s = make_stub(esdf=esdf)
    s._v_max = 0.9
    s._speed_allowance = types.MethodType(FMInferenceBase._speed_allowance, s)
    parallel_ok, scalar_ok = True, True
    for heading in ([1.0, 0.0], [1.0, 1.0], [0.3, -0.9], [-0.5, 0.2]):
        vel = np.array(heading, dtype=float)
        v_allow, _ = s._speed_allowance(np.zeros(2), vel)
        scalar_ok &= np.isscalar(v_allow) or np.ndim(v_allow) == 0
        capped = vel * (v_allow / np.linalg.norm(vel))
        cross = abs(float(np.cross(vel / np.linalg.norm(vel),
                                   capped / np.linalg.norm(capped))))
        parallel_ok &= cross < 1e-9
    check("cap is a scalar", scalar_ok, True)
    check("capped velocity stays parallel (direction untouched)",
          parallel_ok, True)
    # ... and the call site must not add an escape either.
    blk = inspect.getsource(FMInferenceBase._publish_cmd)
    blk = blk[blk.index("Clearance-proportional speed limit"):]
    check("speed-limit call site escape-free", "_enter_escape" in blk, False)
    # The abort path lands; it must not enter escape to "recover".
    src = inspect.getsource(FMInferenceBase.run_sequence)
    check("blind abort lands instead of escaping",
          "_abort_reason" in src and "_enter_escape" not in src, True)


if __name__ == "__main__":
    print("=" * 70)
    print("Anti-collision hardening — replay of the 2026-07-27 crashes")
    print("=" * 70)
    test_cost_gate()
    test_post_check()
    test_speed_limit()
    test_lookahead()
    test_no_escape()
    print("\n" + "=" * 70)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    sys.exit(0)
