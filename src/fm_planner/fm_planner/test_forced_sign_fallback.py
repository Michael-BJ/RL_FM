import logging, sys, os
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.expanduser(
    '~/neo_planner_ws/src/fm_planner/fm_planner'))
import homotopy_frontend as H

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


class _Origin:
    x, y = -2.0, -6.0


class MockESDF:
    map_resolution = 0.1
    map_origin = _Origin()

    def __init__(self):
        rows = int(12.0 / self.map_resolution)
        cols = int(26.0 / self.map_resolution)
        occ = np.zeros((rows, cols), bool)
        c0 = int((7.8 - self.map_origin.x) / self.map_resolution)
        c1 = int((8.2 - self.map_origin.x) / self.map_resolution)
        r0 = int((-1.3 - self.map_origin.y) / self.map_resolution)
        r1 = int((1.3 - self.map_origin.y) / self.map_resolution)
        occ[r0:r1, c0:c1] = True
        self.esdf_map = ndimage.distance_transform_edt(~occ) * self.map_resolution

    def is_ready(self):
        return True


class SpyPlanner:
    """Stands in for MinJerkPlanner: its batch_plan is the stock fallback.
    Records which `map` object topo_batch_plan handed it."""
    init_wpts_num, v_max, T_min, T_max = 2, 1.0, 0.5, 5.0

    def __init__(self):
        self.received_map = None

    def batch_plan(self, m, head, tail):
        self.received_map = m
        self.int_wpts = np.zeros((2, 2))
        self.ts = np.array([1.0, 1.0, 1.0])
        self.final_cost = 0.0
        self.all_solutions = []


esdf = MockESDF()
head = (np.array([0.0, 0.0]), np.zeros(2), np.zeros(2))
tail = (np.array([16.0, 0.0]), np.zeros(2), np.zeros(2))

fails = 0

# ── Case A: forced leg, topo produces NO seed -> fallback MUST get biased ESDF
fe = H.HomotopyFrontEnd()
fe.find_paths = lambda *a, **k: []          # force the empty-seed fallback
fe.forced_sign = 1
sp = SpyPlanner()
H.topo_batch_plan(sp, esdf, head, tail, frontend=fe, fallback=sp.batch_plan)
if isinstance(sp.received_map, H._BiasedESDF):
    print("PASS A: forced fallback received _BiasedESDF (wrong side repelled)")
else:
    print(f"FAIL A: forced fallback got {type(sp.received_map).__name__}, "
          f"expected _BiasedESDF"); fails += 1
if fe.fallback_forced_count == 1:
    print("PASS A2: forced-fallback counter incremented + logged")
else:
    print(f"FAIL A2: fallback_forced_count={fe.fallback_forced_count}"); fails += 1

# Verify the biased grid actually zeroed the wrong (right, sign=-1) side near
# the box -- i.e. the fallback's collision cost really would repel it.
bmap = sp.received_map
d_left  = bmap.get_edt_dis(np.array([8.0,  1.6]))   # forced side (+1): open
d_right = bmap.get_edt_dis(np.array([8.0, -1.6]))   # wrong side (-1): walled
if d_right < 1e-6 and d_left > 0.1:
    print(f"PASS A3: biased grid solid on wrong side "
          f"(d_left={d_left:.2f} open, d_right={d_right:.2f} solid)")
else:
    print(f"FAIL A3: d_left={d_left:.2f} d_right={d_right:.2f}"); fails += 1

# ── Case B: no forcing -> fallback MUST get the raw esdf (behavior unchanged)
fe2 = H.HomotopyFrontEnd()
fe2.find_paths = lambda *a, **k: []
fe2.forced_sign = None
sp2 = SpyPlanner()
H.topo_batch_plan(sp2, esdf, head, tail, frontend=fe2, fallback=sp2.batch_plan)
if sp2.received_map is esdf:
    print("PASS B: unforced fallback received the RAW esdf (unchanged)")
else:
    print(f"FAIL B: unforced fallback got {type(sp2.received_map).__name__}"); fails += 1
if fe2.fallback_forced_count == 0 and fe2.fallback_count == 1:
    print("PASS B2: counters correct for unforced fallback")
else:
    print(f"FAIL B2: forced={fe2.fallback_forced_count} total={fe2.fallback_count}"); fails += 1

print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)"))
sys.exit(1 if fails else 0)
