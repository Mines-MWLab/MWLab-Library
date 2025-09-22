from functools import partial
from pathlib import Path
import numpy as np
import lnoi400
import gdsfactory as gf
import sys, os

# add repo root to path
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import devices, TAUdevices

# --------------------------------------------------------------------------------------
# TAU_Designs_c2: layout study placing exactly one copy of three devices on a chip frame
#   - PM_MLL_cavity_AQ
#   - AM_MLL_cavity_AQ
#   - tunable_mzm_laser_Redwan
# Edge couplers and routing are INCLUDED AS COMMENTS for later enablement.
# --------------------------------------------------------------------------------------

gf.clear_cache()

# utility

def to_itype(points, dbu):
    return [(int(round(x / dbu)), int(round(y / dbu))) for x, y in points]

# database unit (um)
dbu = 0.001

# chip frame
@gf.cell
def chip_frame():
    # size is (width, height); center is chip origin offset
    return gf.get_component("chip_frame", size=(10_000, 5000), center=(5050, 2525))

chip_layout = chip_frame()

# global parameters
input_ext = 10
# double_taper is defined but not used now (kept for when we enable edge couplers)
double_taper = gf.get_component("double_linear_inverse_taper", input_ext=input_ext)
routing_roc = 50.0
frame = 50

# minimum required vertical spacing (includes distance to top/bottom facets)
MIN_SPACING = 490.0  # um

# -----------------------------------------------------------------------------
# Instantiate the three devices (single copies)
# -----------------------------------------------------------------------------
PM = TAUdevices.PM_MLL_cavity_AQ()
AM = TAUdevices.AM_MLL_cavity_AQ()
TZ = TAUdevices.tunable_mzm_laser_Redwan()

@gf.cell
def die_assembled_c2(pitch: float = MIN_SPACING) -> gf.Component:
    """
    Places one copy of each device on the chip, vertically stacked and centered.
    The vertical separation between device centers is `pitch` (>= MIN_SPACING).

    Device order from bottom to top: PM, AM, TZ.
    """
    if pitch < MIN_SPACING:
        raise ValueError(f"pitch must be >= MIN_SPACING ({MIN_SPACING} µm). Got {pitch}.")

    c = gf.Component(name="TAU_Designs_c2_die")
    c << chip_layout

    # Chip extents
    W = chip_layout.dxmax
    H = chip_layout.dymax

    # Compute candidate Y center positions (ensure margins >= pitch)
    # We place three devices centered around H/2 with spacing = pitch.
    total_height = 2 * pitch  # distance between bottom and top centers
    if H < (total_height + 2 * pitch):
        # Conservative check mirroring c1 logic: outer margins >= pitch
        raise RuntimeError(
            f"Chip height {H} µm is too small for 3 devices with pitch {pitch} µm and margins."
        )

    y_center = H / 2.0 - 1500
    y_positions = [y_center - pitch, y_center, y_center + pitch]

    # Horizontal centering for all devices
    x_left = W / 2.0  # we'll center by moving left edge so that component is centered

    # Routing bend factory (kept for later when enabling couplers)
    routing_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
    )

    def place_and_center(ref_comp: gf.Component, y: float, label: str) -> gf.ComponentReference:
        r = c << ref_comp
        r.dmovex(W / 2 - r.xsize / 2)
        r.dmovey(y)
        c.add_label(text=label, position=(r.center[0], r.center[1] + 60), layer=(66, 0))
        return r

    pm_ref = place_and_center(PM, y_positions[0], "PM_MLL_cavity_AQ")
    am_ref = place_and_center(AM, y_positions[1], "AM_MLL_cavity_AQ")
    tz_ref = place_and_center(TZ, y_positions[2], "tunable_mzm_laser_Redwan")

    # ------------------------------------------------------------------
    # EDGE COUPLERS AND ROUTING
    # ------------------------------------------------------------------
    # The code below places left/right edge couplers for each device and routes to
    # their ports using xs_rwg1000/straight_rwg1000. Uncomment to enable.

    # # Left / Right edge-couplers for PM
    # ec_in_pm = c << double_taper
    # ec_in_pm.dmove(ec_in_pm.ports["o1"].dcenter, [-input_ext, pm_ref.ports["o1"].center[1]])
    # ec_out_pm = c << double_taper
    # ec_out_pm.drotate(180)
    # ec_out_pm.dmove(ec_out_pm.ports["o1"].dcenter, [input_ext + W, pm_ref.ports["o2"].center[1]])
    # gf.routing.route_single(c, ec_in_pm.ports["o2"], pm_ref.ports["o1"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")
    # gf.routing.route_single(c, ec_out_pm.ports["o2"], pm_ref.ports["o2"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")

    # # Left / Right edge-couplers for AM
    # ec_in_am = c << double_taper
    # ec_in_am.dmove(ec_in_am.ports["o1"].dcenter, [-input_ext, am_ref.ports["o1"].center[1]])
    # ec_out_am = c << double_taper
    # ec_out_am.drotate(180)
    # ec_out_am.dmove(ec_out_am.ports["o1"].dcenter, [input_ext + W, am_ref.ports["o2"].center[1]])
    # gf.routing.route_single(c, ec_in_am.ports["o2"], am_ref.ports["o1"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")
    # gf.routing.route_single(c, ec_out_am.ports["o2"], am_ref.ports["o2"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")

    # # Left / Right edge-couplers for TZ
    # ec_in_tz = c << double_taper
    # ec_in_tz.dmove(ec_in_tz.ports["o1"].dcenter, [-input_ext, tz_ref.ports["o1"].center[1]])
    # ec_out_tz = c << double_taper
    # ec_out_tz.drotate(180)
    # ec_out_tz.dmove(ec_out_tz.ports["o1"].dcenter, [input_ext + W, tz_ref.ports["o2"].center[1]])
    # gf.routing.route_single(c, ec_in_tz.ports["o2"], tz_ref.ports["o1"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")
    # gf.routing.route_single(c, ec_out_tz.ports["o2"], tz_ref.ports["o2"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")

    return c

# --- Build, FLATTEN, and write ----------------------------------------------
die = die_assembled_c2(pitch=MIN_SPACING)

# make a copy and flatten so we don't mutate the hierarchical source
die_flat = gf.Component("TAU_Designs_c2_die_flat")
die_flat << die
die_flat.flatten()  # pulls all refs up to top-level polygons

die_flat.plot()
die_flat.show()
_ = die_flat.write_gds(gdsdir=Path.cwd())