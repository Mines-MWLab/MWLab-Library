from functools import partial
from pathlib import Path
import numpy as np
import lnoi400
import gdsfactory as gf
import sys, os

# -----------------------------------------------------------------------------
# Repo root on path and constants
# -----------------------------------------------------------------------------
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import devices, components

gf.clear_cache()

# -----------------------------------------------------------------------------
# Edge coupler: load from GDS
# -----------------------------------------------------------------------------
# Path to edge-coupler GDS (relative to this file)
EDGE_COUPLER_GDS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "utility_files", "LXT_LT_edge_coupler.gds")
)
EDGE_COUPLER_CELL = None  # set to the actual cell name if the topcell isn't the coupler

EDGE_COUPLER = gf.import_gds(EDGE_COUPLER_GDS, cellname=EDGE_COUPLER_CELL)
EDGE_COUPLER_775 = components.double_taper_edge_coupler_JL()

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def to_itype(points, dbu):
    return [(int(round(x / dbu)), int(round(y / dbu))) for x, y in points]

dbu = 0.001

# -----------------------------------------------------------------------------
# Chip frame
# -----------------------------------------------------------------------------
@gf.cell
def chip_frame():
    return gf.get_component("chip_frame", size=(10_000, 5000), center=(5050, 2525))

chip_layout = chip_frame()

# -----------------------------------------------------------------------------
# Global parameters
# -----------------------------------------------------------------------------
input_ext = 5.0
routing_roc = 50.0

# Coupler cross-section is 0.9 µm
COUPLER_CROSS_SECTION = "xs_rwg900"
COUPLER_CROSS_SECTION_775 = "xs_rwg2000"
TAPER_LENGTH = 300.0

# Spacing constraints
MIN_SPACING_SAME = 490.0  # µm (same cross-section rows)
EDGE_MARGIN      = 400.0  # µm (top/bottom clearance)
SPACING_DIFF     = 180.0  # µm (between adjacent different cross-sections)

# Three row cross-sections
CROSS_SECTIONS = ["xs_rwg3000", "xs_rwg2750", "xs_rwg2500"]
STRAIGHT_NAMES = ["straight_rwg3000", "straight_rwg2750", "straight_rwg2500"]

# Base straight length
OPA_LENGTH = 8000.0

# -----------------------------------------------------------------------------
# Pre-build per-set straights and transition tapers 
# -----------------------------------------------------------------------------
OPA_STRAIGHTS_WITH_MMI = [
    devices.OPA_straight_waveguide(length=OPA_LENGTH, cross_section=cs, with_mmi=True)
    for cs in CROSS_SECTIONS
]

OPA_STRAIGHTS_SINGLE = [
    devices.OPA_straight_waveguide(length=OPA_LENGTH, cross_section=cs, with_mmi=False)
    for cs in CROSS_SECTIONS
]

TRANSITION_TAPERS = [
    gf.components.taper_cross_section(
        cross_section1=COUPLER_CROSS_SECTION,  # 0.9 µm side (edge-coupler)
        cross_section2=cs,                     # row cross-section
        length=TAPER_LENGTH,
        linear=True,
    )
    for cs in CROSS_SECTIONS
]

TRANSITION_TAPERS_775 = [
    gf.components.taper_cross_section(
        cross_section1=COUPLER_CROSS_SECTION_775,
        cross_section2=cs,
        length=TAPER_LENGTH-200,
        linear=True,
    )
    for cs in CROSS_SECTIONS
]

# Number of rows (waveguides) that keep the dual-input MMI variant.
# We want 4 sets × 3 cross-section bands = 12 rows.
ROWS_WITH_MMI = 12

# -----------------------------------------------------------------------------
# Main builder with interleaved packing (0 -> 1 -> 2 -> repeat)
# -----------------------------------------------------------------------------
@gf.cell
def die_assembled_grouped(
    min_spacing_same: float = MIN_SPACING_SAME,  # 490 µm for same cross-section
    edge_margin: float = EDGE_MARGIN,            # 400 µm margins
    spacing_diff: float = SPACING_DIFF,          # 180 µm between different cross-sections
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
    y_min = edge_margin + 50.0
    y_max = H_total - edge_margin
    if y_max <= y_min:
        raise RuntimeError("Edge margins exceed chip height.")

    # Routing bend factory
    routing_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
    )

    # Interleaved greedy packer:
    # Place rows as close as allowed while cycling types 0->1->2 and honoring:
    # - y >= prev_y + spacing_diff           (adjacent rows diff spacing)
    # - y >= last_y_for_type + min_spacing_same  (repeat type spacing)
    def pack_interleaved(y0: float, y1: float) -> list[tuple[float, int]]:
        last_for_type = {0: None, 1: None, 2: None}
        seq = []
        prev_y = None
        t = 0  # type index: 0 -> 1 -> 2 -> repeat

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
            t = (t + 1) % 3

        return seq

    # Compute placements across the full usable height
    placements = pack_interleaved(y_min, y_max)

    # Place and route each row
    for row_idx, (y_center, set_idx) in enumerate(placements):
        # set-specific parts
        use_mmi = row_idx < ROWS_WITH_MMI

        opa_straight = (
            OPA_STRAIGHTS_WITH_MMI[set_idx]
            if use_mmi
            else OPA_STRAIGHTS_SINGLE[set_idx]
        )
        transition_taper = TRANSITION_TAPERS[set_idx]
        transition_taper_775 = TRANSITION_TAPERS_775[set_idx]
        xs_name = CROSS_SECTIONS[set_idx]
        straight_name = STRAIGHT_NAMES[set_idx]

        # straight
        opa_ref = c << opa_straight
        opa_ref.dmovex(W_total / 2.0 - opa_ref.xsize / 2.0)
        opa_ref.dmovey(y_center - opa_ref.ysize / 2.0)

        # input edge couplers + transition tapers 
        if use_mmi:
            INPUT_SPACING = 125.0

            def add_input_route(
                target_port_name: str,
                vertical_offset: float,
            ) -> tuple[gf.ComponentReference, gf.ComponentReference] | None:
                if target_port_name not in opa_ref.ports:
                    return None

                target_port = opa_ref.ports[target_port_name]

                desired_y = target_port.center[1] + vertical_offset
                effective_offset = vertical_offset

                if target_port_name == "in_top":
                    coupler_component = EDGE_COUPLER_775
                    taper_component = transition_taper_775
                else:
                    coupler_component = EDGE_COUPLER
                    taper_component = transition_taper

                ec = c << coupler_component
                ec.dmove(
                    ec.ports["o1"].dcenter,
                    [-input_ext, target_port.center[1] + effective_offset],
                )
                taper = c << taper_component
                taper.connect("o1", ec.ports["o2"])

                start_port = taper.ports["o2"]

                def bend_factory(*, size, cross_section=xs_name, **_):
                    return lnoi400.cells.bend_S_spline_varying_width(
                        size=size,
                        cross_section1=cross_section,
                        cross_section2=cross_section,
                        npoints=201,
                    )

                gf.routing.route_single_sbend(
                    c,
                    port1=start_port,
                    port2=target_port,
                    bend_s=bend_factory,
                    cross_section=xs_name,
                    allow_width_mismatch=True
                )
                return ec, taper

            half_spacing = INPUT_SPACING / 2.0
            add_input_route("in_bot", -half_spacing)
            add_input_route("in_top", half_spacing)
        else:
            # Single-input configuration (legacy straight without MMI)
            try:
                single_input_port = opa_ref.ports["o1"]
            except KeyError:
                single_input_port = None
            if single_input_port is not None:
                ec_single = c << EDGE_COUPLER
                ec_single.dmove(
                    ec_single.ports["o1"].dcenter,
                    [-input_ext, single_input_port.center[1]],
                )
                taper_single = c << transition_taper
                taper_single.connect("o1", ec_single.ports["o2"])

                gf.routing.route_single(
                    c,
                    taper_single.ports["o2"],
                    single_input_port,
                    cross_section=xs_name,
                    bend=routing_bend,
                    straight=straight_name,
                    allow_width_mismatch=True
                )

        # output edge coupler + transition taper 
        ec_out = c << EDGE_COUPLER
        ec_out.drotate(180)
        ec_out.dmove(
            ec_out.ports["o1"].dcenter,
            [input_ext + W_total, opa_ref.ports["o2"].center[1]],
        )
        taper_out = c << transition_taper
        taper_out.drotate(180)
        taper_out.connect("o1", ec_out.ports["o2"])

        # route output (use per-set cross-section + straight recipe)
        gf.routing.route_single(
            c,
            taper_out.ports["o2"],
            opa_ref.ports["o2"],
            cross_section=xs_name,
            bend=routing_bend,
            straight=straight_name,
            allow_width_mismatch=True
        )

    # Add crux made of two 4 µm wide, 250 µm long straight waveguides.
    crux = gf.Component("crux")
    horiz = crux << gf.components.rectangle(size=(250.0, 4.0), layer=(4, 0))
    horiz.move((-125.0, -2.0))
    vert = crux << gf.components.rectangle(size=(4.0, 250.0), layer=(4, 0))
    vert.move((-2.0, -125.0))

    crux_ref = c << crux
    left_facet_x = chip_layout.xmin
    right_facet_x = chip_layout.xmax
    top_facet_y = chip_layout.ymax
    bot_facet_y = chip_layout.ymin
    crux_center_x = left_facet_x + 50.0 + 1500.0  # 1375 µm offset + half length (125 µm)
    crux_center_y = top_facet_y - 50.0 - 126.0    # 1 µm clearance + half length (125 µm)
    crux_ref.move((crux_center_x, crux_center_y))
    crux_ref_mirror = c << crux
    crux_ref_mirror.move((right_facet_x - crux_center_x, crux_center_y))
    crux_ref_mirror1 = c << crux
    crux_ref_mirror1.move((crux_center_x, 50.0 + 126.0))
    crux_ref_mirror2 = c << crux
    crux_ref_mirror2.move((right_facet_x-crux_center_x, 50.0 + 126.0 ))

    return c

# -----------------------------------------------------------------------------
# Build and show die
# -----------------------------------------------------------------------------
die = die_assembled_grouped()

# Work on a flattened copy for export so downstream translations remain robust.
die_flat = gf.Component("TAU_Designs_c1_die_flat")
die_flat << die
die_flat.flatten()

die_flat.plot()
die_flat.show()
# _ = die_flat.write_gds(gdsdir=Path.cwd())
