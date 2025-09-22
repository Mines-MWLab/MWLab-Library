
from functools import partial
from pathlib import Path
import numpy as np
import lnoi400
import gdsfactory as gf
import sys, os

# add repo root to path
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import devices

gf.clear_cache()

# utility
def to_itype(points, dbu):
    return [(int(round(x / dbu)), int(round(y / dbu))) for x, y in points]

dbu = 0.001

# chip frame
@gf.cell
def chip_frame():
    return gf.get_component("chip_frame", size=(10_000, 5000), center=(5050, 2525))

chip_layout = chip_frame()

# global parameters
input_ext = 10
double_taper = gf.get_component("double_linear_inverse_taper", input_ext=input_ext)
routing_roc = 50.0

COUPLER_CROSS_SECTION = "xs_rwg1000"
TAPER_LENGTH = 50.0

# --- Constraints from your spec ---
MIN_SPACING_SAME = 490.0   # µm, between waveguides of the same width
EDGE_MARGIN      = 350.0   # µm, clearance to top and bottom chip edges

# For now: use 3 µm cross-section everywhere 
CROSS_SECTIONS = ["xs_rwg3000", "xs_rwg3000", "xs_rwg2500"]
STRAIGHT_NAMES = ["straight_rwg3000", "straight_rwg3000", "straight_rwg2500"]

# Base straight length
OPA_LENGTH = 400.0

# Pre-build a straight OPA for each set (one per cross-section)
OPA_STRAIGHTS = [
    devices.OPA_straight_waveguide(length=OPA_LENGTH, cross_section=cs)
    for cs in CROSS_SECTIONS
]

TRANSITION_TAPERS = [
    gf.components.taper_cross_section(
        cross_section1=COUPLER_CROSS_SECTION,
        cross_section2=cs,
        length=TAPER_LENGTH,
        linear=True,
    )
    for cs in CROSS_SECTIONS
]

@gf.cell
def die_assembled_grouped(
    min_spacing_same: float = MIN_SPACING_SAME,  # 490 µm for same cross-section
    edge_margin: float = EDGE_MARGIN,            # 350 µm top/bottom margins
    spacing_diff: float = 150.0,                  # 20 µm between different cross-sections
) -> gf.Component:
    """
    Build a die with interleaved rows of three cross-sections (0->1->2->repeat).
    Enforce: >= spacing_diff between adjacent (different) rows and
             >= min_spacing_same between rows of the same cross-section.
    Keep edge_margin at top and bottom.
    """
    c = gf.Component()
    c << chip_layout

    H_total = chip_layout.dymax
    W_total = chip_layout.dxmax

    # Usable vertical range after top/bottom edge margins
    y_min = edge_margin
    y_max = H_total - edge_margin
    if y_max <= y_min:
        raise RuntimeError("Edge margins exceed chip height.")

    # Routing bend factory (unchanged)
    routing_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
    )

    # Helper: interleave types [0,1,2,0,1,2,...] while meeting spacing constraints.
    # Greedy: for the next type, place the smallest y that satisfies both:
    #   (1) y >= prev_y + spacing_diff  (adjacent rows)
    #   (2) y >= last_y_for_this_type + min_spacing_same  (same-type separation)
    def pack_interleaved(y0: float, y1: float) -> list[tuple[float, int]]:
        last_for_type = {0: None, 1: None, 2: None}
        seq = []
        prev_y = None
        t = 0  # start with type 0 -> 1 -> 2 -> repeat

        while True:
            needs = [y0]
            if prev_y is not None:
                needs.append(prev_y + spacing_diff)
            if last_for_type[t] is not None:
                needs.append(last_for_type[t] + min_spacing_same)
            y = max(needs)

            if y > y1:
                break

            seq.append((y, t))
            last_for_type[t] = y
            prev_y = y
            t = (t + 1) % 3  # cycle 0->1->2

        return seq

    # Generate (y, type_idx) placements across full usable height
    placements = pack_interleaved(y_min, y_max)

    # Place rows
    for y_center, set_idx in placements:
        # select set-specific parts
        opa_straight = OPA_STRAIGHTS[set_idx]
        transition_taper = TRANSITION_TAPERS[set_idx]
        xs_name = CROSS_SECTIONS[set_idx]
        straight_name = STRAIGHT_NAMES[set_idx]

        # straight
        opa_ref = c << opa_straight
        opa_ref.dmovex(W_total / 2.0 - opa_ref.xsize / 2.0)
        opa_ref.dmovey(y_center - opa_ref.ysize / 2.0)

        # input coupler + transition taper
        ec_in = c << double_taper
        ec_in.dmove(
            ec_in.ports["o1"].dcenter,
            [-input_ext, opa_ref.ports["o1"].center[1]],
        )
        taper_in = c << transition_taper
        taper_in.connect("o1", ec_in.ports["o2"])

        # output coupler + transition taper
        ec_out = c << double_taper
        ec_out.drotate(180)
        ec_out.dmove(
            ec_out.ports["o1"].dcenter,
            [input_ext + W_total, opa_ref.ports["o2"].center[1]],
        )
        taper_out = c << transition_taper
        taper_out.drotate(180)
        taper_out.connect("o1", ec_out.ports["o2"])

        # routes (use the per-set cross-section + straight)
        gf.routing.route_single(
            c,
            taper_in.ports["o2"],
            opa_ref.ports["o1"],
            cross_section=xs_name,
            bend=routing_bend,
            straight=straight_name,
        )
        gf.routing.route_single(
            c,
            taper_out.ports["o2"],
            opa_ref.ports["o2"],
            cross_section=xs_name,
            bend=routing_bend,
            straight=straight_name,
        )

    return c

# --- Build and show die -------------------------------------------------------
die = die_assembled_grouped()
die.plot()
die.show()
# _ = die.write_gds(gdsdir=Path.cwd())
