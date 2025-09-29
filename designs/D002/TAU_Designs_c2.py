from functools import partial
from pathlib import Path
import math
import numpy as np
import lnoi400
import gdsfactory as gf
import sys, os

# add repo root to path
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import devices, TAUdevices, components
gf.config.write_ports_on_component = True

# --------------------------------------------------------------------------------------
# TAU_Designs_c2: layout study placing exactly one copy of three devices on a chip frame
#   - PM_MLL_cavity_AQ
#   - AM_MLL_cavity_AQ
#   - tunable_mzm_laser_Redwan
# Edge couplers and routing are INCLUDED AS COMMENTS for later enablement.
# --------------------------------------------------------------------------------------

gf.clear_cache()

# Edge coupler (reuse the same as TAU_Designs_c1)
EDGE_COUPLER_GDS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "utility_files", "LXT_LT_edge_coupler.gds")
)
EDGE_COUPLER_CELL = None  # set explicitly if the edge coupler cell is not the top cell
EDGE_COUPLER = gf.import_gds(EDGE_COUPLER_GDS, cellname=EDGE_COUPLER_CELL)

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
input_ext = 5
# edge coupler component (referenced in legacy code comments)
double_taper = EDGE_COUPLER
routing_roc = 50.0
frame = 50

# minimum required vertical spacing (includes distance to top/bottom facets)
MIN_SPACING = 490.0  # um

# per-device manual placement tweaks (dx, dy) in microns
DEVICE_OFFSETS = {
    "PM_MLL_cavity_AQ": (80.0, -150.0),
    "AM_MLL_cavity_AQ": (80.0, -200.0),
    "tunable_mzm_laser_Redwan": (770.0, -100.0),
    "soliton_ring_Redwan": (2000.0, 0.0),
}

# Spiral panel global offset (dx, dy) in microns
SPIRAL_SWEEP_OFFSET = (-4000.0, -3400.0)
SPIRAL_BLOCK_SEPARATION = 500.0
SPIRAL_VORTEX_PORT_PITCH = 25.0

# Rings panel global offset (dx, dy) in microns
RING_VORTEX_PORT_PITCH = 125.0
RING_VORTEX_SWEEP_OFFSET = (300.0, 4300.0)
RING_VORTEX_SWEEP_Q_VALUES = [336, 338, 340]  # top to bottom
RING_VORTEX_SWEEP_GAPS = [0.30, 0.40, 0.50, 0.60]   
RING_VORTEX_SWEEP_ROW_PITCH = 500.0
RING_VORTEX_SWEEP_COL_PITCH = 125.0

# soliton ring sweep configuration
SOLITON_RING_SWEEP_COUPLING_GAPS = [2.683, 2.803, 2.908, 3.048, 3.261]  # µm (2683–3261 nm)
SOLITON_RING_SWEEP_INNER_OUTER_GAPS = [6.1, 6.3, 6.5]  # µm
SOLITON_RING_SWEEP_OFFSET = (3104.5, -1000.0)
SOLITON_RING_SWEEP_HORIZONTAL_PITCH = 600.0
SOLITON_RING_SWEEP_VERTICAL_PITCH = 25.0
SOLITON_RING_SWEEP_BLOCK_VERTICAL_SPACING = 650.0

# NanoPh EO phase shifter placement offset (dx, dy) in microns
NANOPH_EO_PS_OFFSET = (-1000.0, 1800.0)

# NanoPh grating structure placement controls
NANOPH_GRATING_OFFSET = (-1100.0, 100.0)
NANOPH_GRATING_PERIODS_TOP = [1.00, 1.05, 1.10, 1.15]
NANOPH_GRATING_PERIOD_STEP = 0.05
NANOPH_GRATING_ROW_PITCH = 400.0
NANOPH_GRATING_COL_PITCH = 450.0


def build_spiral_sweep_panel() -> gf.Component:
    panel = gf.Component("spiral_vortex_sweep_panel")

    notch_categories = ("S", "C_in", "C_out")

    def make_q_list(values: list[int]) -> list[list[int]]:
        return [values[:] for _ in range(3)]

    spiral_blocks = [
        {
            "base_y": 5000.0,
            "x_offset": 74.4,
            "y_offset": 74.0,
            "loops": [1, 1, 1],
            "q_values": make_q_list([331, 332, 333, 334, 335]),
            "w_notch": {"S": 0.25, "C_in": 0.3, "C_out": 0.3},
            "variable": [False, False, False],
        },
        {
            "base_y": 5000.0 - SPIRAL_BLOCK_SEPARATION,
            "x_offset": 74.4,
            "y_offset": 74.0,
            "loops": [1, 1, 1],
            "q_values": make_q_list([331, 332, 333, 334, 335]),
            "w_notch": {"S": 0.25, "C_in": 0.3, "C_out": 0.3},
            "variable": [True, True, True],
        },
        {
            "base_y": 5000.0 - 2 * SPIRAL_BLOCK_SEPARATION,
            "x_offset": 71.4,
            "y_offset": 71.0,
            "loops": [2, 2, 2],
            "q_values": make_q_list([649, 651, 653, 655, 657]),
            "w_notch": {"S": 0.25, "C_in": 0.3, "C_out": 0.3},
            "variable": [False, False, False],
        },
        {
            "base_y": 5000.0 - 3 * SPIRAL_BLOCK_SEPARATION,
            "x_offset": 68.4,
            "y_offset": 68.0,
            "loops": [3, 3, 3],
            "q_values": make_q_list([953, 956, 959, 962, 965]),
            "w_notch": {"S": 0.25, "C_in": 0.3, "C_out": 0.3},
            "variable": [False, False, False],
        },
    ]

    x_step = 125.0

    for block_index, block in enumerate(spiral_blocks):
        for idx in range(15):
            i = idx + 1
            category = idx // 5
            category_index = idx % 5
            notch_type = notch_categories[category]
            q_value = block["q_values"][category][category_index]
            loops_spiral = block["loops"][category]
            w_notch = block["w_notch"][notch_type]
            variable = block["variable"][category]

            spiral = panel.add_ref(
                TAUdevices.spiral_vortex_beam_emitter_equal_arc_spacing_AC(
                    q=q_value,
                    W_notch=w_notch,
                    notch_type=notch_type,
                    variable_pillar_dist=variable,
                    loops_spiral=loops_spiral,
                )
            )
            spiral.drotate(270)
            target_center_x = (i + 1) * x_step - block["x_offset"]
            base_center_y = block["base_y"] - block["y_offset"]

            spiral.dmovex(target_center_x - spiral.center[0])
            spiral.dmovey(base_center_y - spiral.center[1])

            if "o1" in spiral.ports:
                desired_port_y = base_center_y - (i - 1) * SPIRAL_VORTEX_PORT_PITCH
                current_port_y = spiral.ports["o1"].center[1]
                spiral.dmovey(desired_port_y - current_port_y)
            port_name = f"sp_b{block_index}_i{idx+1}"
            panel.add_port(name=port_name, port=spiral.ports["o1"])

    return panel


def build_ring_vortex_sweep(
    coupler_x_position: float,
    routing_roc: float,
) -> gf.Component:
    panel = gf.Component("ring_vortex_sweep_panel")

    routing_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
        cross_section="xs_rwg900",
    )

    total_columns = len(RING_VORTEX_SWEEP_GAPS)

    for row, q_value in enumerate(RING_VORTEX_SWEEP_Q_VALUES):
        base_center_y = -row * RING_VORTEX_SWEEP_ROW_PITCH
        for col, gap in enumerate(RING_VORTEX_SWEEP_GAPS):
            ring = panel.add_ref(
                TAUdevices.ring_vortex_beam_emitter_AC(
                    q=q_value,
                    W_gap=gap,
                    notch_type="C_in",
                )
            )
            ring.drotate(180)
            target_center_x = col * RING_VORTEX_SWEEP_COL_PITCH + row * 4 * RING_VORTEX_SWEEP_COL_PITCH
            ring.dmovex(target_center_x - ring.center[0])
            ring.dmovey(base_center_y - ring.center[1])

            in_port = ring.ports["o1"] if "o1" in ring.ports else None
            if in_port is not None:
                desired_port_y = base_center_y + (3 - col) * RING_VORTEX_PORT_PITCH
                current_port_y = in_port.center[1]
                ring.dmovey(desired_port_y - current_port_y)
                in_port = ring.ports["o1"]

                input_coupler = panel << EDGE_COUPLER
                input_coupler.drotate(0)
                input_coupler.dmove(
                    input_coupler.ports["o1"].dcenter,
                    (coupler_x_position, in_port.center[1]),
                )

                in_taper = panel.add_ref(
                    gf.components.taper_cross_section(
                        cross_section1="xs_rwg900",
                        cross_section2="xs_rwg800",
                        length=100.0,
                        linear=True,
                    )
                )
                in_taper.connect("o1", input_coupler.ports["o2"])

                gf.routing.route_single(
                    panel,
                    port1=in_taper.ports["o2"],
                    port2=in_port,
                    cross_section="xs_rwg800",
                    bend=routing_bend,
                    radius=routing_roc,
                    straight="straight_rwg800"
                )

            out_port = ring.ports["o2"] if "o2" in ring.ports else None
            if out_port is not None:
                ring_index = row * total_columns + col

                output_coupler = panel << EDGE_COUPLER
                output_coupler.drotate(-90)
                output_coupler.dmove(
                    output_coupler.ports["o1"].dcenter,
                    (ring.ports["o2"].dcenter[0] + routing_roc, chip_layout.dymax + input_ext - RING_VORTEX_SWEEP_OFFSET[1]),
                )

                out_taper = panel.add_ref(
                    gf.components.taper_cross_section(
                        cross_section1="xs_rwg900",
                        cross_section2="xs_rwg800",
                        length=100.0,
                        linear=True,
                    )
                )
                out_taper.connect("o1", output_coupler.ports["o2"])

                gf.routing.route_single(
                    panel,
                    port1=out_taper.ports["o2"],
                    port2=out_port,
                    cross_section="xs_rwg800",
                    bend=routing_bend,
                    radius=routing_roc,
                    straight="straight_rwg800"
                )

    return panel

# -----------------------------------------------------------------------------
# Instantiate the three devices (single copies)
# -----------------------------------------------------------------------------
PM = TAUdevices.PM_MLL_cavity_AQ(modulation_length=9150.0)
AM = TAUdevices.AM_MLL_cavity_AQ(modulation_length=8440.0)
TZ = TAUdevices.tunable_mzm_laser_Redwan()
NANOPH_PS = TAUdevices.NanoPh_eo_phase_shifter()
NANOPH_GRATING = TAUdevices.nanoph_grating_structure_AC()

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
        dx, dy = DEVICE_OFFSETS.get(label, (0.0, 0.0))
        r = c << ref_comp
        r.dmovex(W / 2 - r.xsize / 2 + dx)
        r.dmovey(y + dy)
        return r

    pm_ref = place_and_center(PM, y_positions[0], "PM_MLL_cavity_AQ")
    am_ref = place_and_center(AM, y_positions[1], "AM_MLL_cavity_AQ")
    tz_ref = place_and_center(TZ, y_positions[2], "tunable_mzm_laser_Redwan")

    # Spiral emitter sweep panel
    spiral_panel = build_spiral_sweep_panel()
    panel_ref = c << spiral_panel
    panel_width = panel_ref.dxmax - panel_ref.dxmin
    panel_height = panel_ref.dymax - panel_ref.dymin
    panel_ref.dmovex(W / 2 - panel_width / 2 + SPIRAL_SWEEP_OFFSET[0])
    panel_ref.dmovey(H / 2 - panel_height / 2 + SPIRAL_SWEEP_OFFSET[1])
    c.add_label(
        text="spiral_vortex_sweep_panel",
        position=(panel_ref.center[0], panel_ref.center[1] + 60),
        layer=(66, 0),
    )

    left_facet_x = chip_layout.dxmin
    spiral_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
        cross_section="xs_rwg900",
    )
    spiral_straight = partial(
        gf.components.straight,
        cross_section="xs_rwg900",
    )
    taper_component = gf.components.taper_cross_section(
        cross_section1="xs_rwg800",
        cross_section2="xs_rwg900",
        length=50.0,
        linear=True,
    )

    def route_spiral_ports_to_left(ref: gf.ComponentReference) -> None:
        if not ref.ports:
            return
        for port in list(ref.ports):
            taper_ref = c << taper_component
            taper_ref.connect("o1", port)

            coupler_ref = c << EDGE_COUPLER
            coupler_ref.drotate(0)
            coupler_ref.dmove(
                coupler_ref.ports["o1"].dcenter,
                (left_facet_x - input_ext, port.center[1]),
            )

            path = gf.Path([taper_ref.ports["o2"].center, coupler_ref.ports["o2"].center])
            c << path.extrude(cross_section="xs_rwg900")

    route_spiral_ports_to_left(panel_ref)

    # ------------------------------------------------------------------
    # NANOPH EO PHASE SHIFTER
    # ------------------------------------------------------------------
    nanoph_ps_ref = c << TAUdevices.NanoPh_eo_phase_shifter(modulation_length=4000.0)
    target_ps_x = W / 2.0 + NANOPH_EO_PS_OFFSET[0]
    target_ps_y = H / 2.0 + NANOPH_EO_PS_OFFSET[1]
    nanoph_ps_ref.dmovex(target_ps_x - nanoph_ps_ref.center[0])
    nanoph_ps_ref.dmovey(target_ps_y - nanoph_ps_ref.center[1])

    top_facet_y = chip_layout.dymax + input_ext
    ps_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
        cross_section="xs_rwg2500",
    )

    taper = gf.components.taper_cross_section(
            cross_section1="xs_rwg900",
            cross_section2="xs_rwg2500",
            length=400.0,
            linear=True,
        )

    for port_name in ("o1", "o2"):
        if port_name not in nanoph_ps_ref.ports:
            continue
        port = nanoph_ps_ref.ports[port_name]
        coupler_ref = c << EDGE_COUPLER
        coupler_ref.drotate(-90)
        coupler_ref.dmove(
            coupler_ref.ports["o1"].dcenter,
            (port.center[0] + (-1 if port_name == "o1" else 1) * routing_roc , top_facet_y),
        )
        
        taper_ref = c << taper
        taper_ref.connect("o1", coupler_ref.ports["o2"])

        gf.routing.route_single(
            c,
            port1=taper_ref["o2"],
            port2=port,
            cross_section="xs_rwg2500",
            bend=ps_bend,
            radius=routing_roc,
            straight="straight_rwg2500",
            allow_width_mismatch=True
        )

    # ------------------------------------------------------------------
    # NANOPH GRATING STRUCTURE
    # ------------------------------------------------------------------
    top_periods = list(NANOPH_GRATING_PERIODS_TOP)
    step = NANOPH_GRATING_PERIOD_STEP
    bottom_periods = [max(top_periods) + (idx + 1) * step for idx in range(len(top_periods))]

    grating_cols = len(top_periods)
    grating_rows = 2
    base_grating_x = W / 2.0 + NANOPH_GRATING_OFFSET[0]
    base_grating_y = H / 2.0 + NANOPH_GRATING_OFFSET[1]

    for row in range(grating_rows):
        row_offset_y = (row - (grating_rows - 1) / 2.0) * NANOPH_GRATING_ROW_PITCH
        row_periods = top_periods if row == grating_rows - 1 else bottom_periods
        for col, period in enumerate(row_periods):
            grating_ref = c << TAUdevices.nanoph_grating_structure_AC(period=period)

            if row == grating_rows - 1:
                grating_ref.drotate(90)

            col_offset_x = (col - (grating_cols - 1) / 2.0) * NANOPH_GRATING_COL_PITCH
            target_x = base_grating_x + col_offset_x
            target_y = base_grating_y + row_offset_y
            grating_ref.dmovex(target_x - grating_ref.center[0])
            grating_ref.dmovey(target_y - grating_ref.center[1])

            c.add_label(
                text=f"nanoph_grating_AC_p{period:.3f}_r{row}_c{col}",
                position=(grating_ref.center[0], grating_ref.center[1] + 60),
                layer=(66, 0),
            )

    # Place ring vortex sweep panel 
    coupler_x_local = (left_facet_x - input_ext) - RING_VORTEX_SWEEP_OFFSET[0]
    ring_panel = build_ring_vortex_sweep(
        coupler_x_position=coupler_x_local,
        routing_roc=routing_roc,
    )
    ring_panel_ref = c << ring_panel
    ring_panel_ref.dmovex(RING_VORTEX_SWEEP_OFFSET[0])
    ring_panel_ref.dmovey(RING_VORTEX_SWEEP_OFFSET[1])

    # Soliton ring sweep (coupling gap by row, inner/outer gap by column)
    soliton_coupling_gaps = list(SOLITON_RING_SWEEP_COUPLING_GAPS)
    soliton_inner_outer_gaps = list(SOLITON_RING_SWEEP_INNER_OUTER_GAPS)

    if soliton_coupling_gaps and soliton_inner_outer_gaps:
        base_x = W / 2 + SOLITON_RING_SWEEP_OFFSET[0]
        base_y = H / 2 + SOLITON_RING_SWEEP_OFFSET[1]
        soliton_refs: list[tuple[float, float, gf.ComponentReference, int, int]] = []
        for row_idx, coupling_gap in enumerate(soliton_coupling_gaps):
            block_base_y = base_y + row_idx * SOLITON_RING_SWEEP_BLOCK_VERTICAL_SPACING
            for col_idx, inner_outer_gap in enumerate(soliton_inner_outer_gaps):
                ring_component = TAUdevices.soliton_ring_Redwan(
                    coupling_gap=coupling_gap,
                    gap_inner_outer=inner_outer_gap,
                )
                ring_ref = c << ring_component
                target_x = base_x + col_idx * SOLITON_RING_SWEEP_HORIZONTAL_PITCH
                target_y = block_base_y + col_idx * SOLITON_RING_SWEEP_VERTICAL_PITCH
                origin_center = ring_ref.center
                ring_ref.dmovex(target_x - origin_center[0])
                ring_ref.dmovey(target_y - origin_center[1])
                soliton_refs.append((coupling_gap, inner_outer_gap, ring_ref, row_idx, col_idx))
                c.add_label(
                    text=f"soliton_gap{coupling_gap:.3f}_io{inner_outer_gap:.1f}_r{row_idx}_c{col_idx}",
                    position=(ring_ref.center[0], ring_ref.center[1] + 45),
                    layer=(66, 0),
                )

        top_facet_y = chip_layout.dymax
        right_facet_x = chip_layout.dxmax
        ring_bend = partial(
            gf.components.bend_euler,
            radius=routing_roc,
            with_arc_floorplan=True,
            cross_section="xs_rwg1380",
        )
        ring_straight = partial(
            gf.components.straight,
            cross_section="xs_rwg1380",
        )
        taper_ring_component = gf.components.taper_cross_section(
            cross_section1="xs_rwg1380",
            cross_section2="xs_rwg1380",
            length=1.0,
            linear=True,
        )

        for coupling_gap, inner_outer_gap, ring_ref, row_idx, col_idx in soliton_refs:
            if "o1" not in ring_ref.ports:
                continue
            ring_port = ring_ref.ports["o1"]
            taper_ref = c << taper_ring_component
            coupler_ref = c << components.linear_inverse_taper_AQ(
                taper_length=200, 
                input_ext=10, 
                cross_section_end="xs_rwg1380"
                )
            coupler_ref.drotate(-90)
            taper_out_center = taper_ref.ports["o2"].center
            x_west_base = ring_port.center[0] - 100.0 
            x_west = x_west_base - (col_idx) * (SOLITON_RING_SWEEP_HORIZONTAL_PITCH - SOLITON_RING_SWEEP_VERTICAL_PITCH) - 3 * (len(soliton_coupling_gaps) - 1 - row_idx) * SOLITON_RING_SWEEP_VERTICAL_PITCH - 275
            y_north = ring_port.center[1] + 500
            y_west = ring_port.center[1]

            coupler_ref.dmove(
                coupler_ref.ports["o1"].dcenter,
                (x_west, top_facet_y + input_ext),
            )

            taper_ref.connect("o2", coupler_ref.ports["o2"])

            waypoints = [
                    (x_west, y_north),
                    (x_west_base, y_north),
                    (x_west_base, y_west)
            ]
            gf.routing.route_single(
                c,
                port1=taper_ref.ports["o1"],
                port2=ring_port,
                cross_section="xs_rwg1380",
                bend=ring_bend,
                radius=100.0,
                straight=ring_straight,
                waypoints=waypoints,
                allow_width_mismatch=True
            )

            # Route the right-most port to a right-facet edge coupler
            if "o2" in ring_ref.ports:
                right_port = ring_ref.ports["o2"]
                taper_right = c << taper_ring_component
                taper_right.connect("o1", right_port)
                coupler_right = c << components.linear_inverse_taper_AQ(
                taper_length=200, 
                input_ext=10, 
                cross_section_end="xs_rwg1380"
                )
                coupler_right.drotate(180)
                coupler_right.dmove(
                    coupler_right.ports["o1"].dcenter,
                    (right_facet_x + input_ext, right_port.center[1]),
                )
                gf.routing.route_single(
                    c,
                    port1=coupler_right.ports["o2"],
                    port2=taper_right.ports["o2"],
                    cross_section="xs_rwg1380",
                    bend=ring_bend,
                    radius=routing_roc,
                    straight=ring_straight,
                    allow_width_mismatch=True
                )

    # ------------------------------------------------------------------
    # EDGE COUPLERS AND ROUTING
    # ------------------------------------------------------------------
    # TZ Edge Coupler and routing
    if "o1" in tz_ref.ports:
        tz_output_port = tz_ref.ports["o1"]
        tz_ec_top = c << components.tilted_inverse_taper_AQ(taper_length=200, input_ext=12)
        tz_ec_top.drotate(-90)
        top_facet_y = chip_layout.ymax + 2 * input_ext
        tz_ec_top_x = tz_output_port.center[0] - routing_roc
        tz_ec_top.dmove(
            tz_ec_top.ports["o2"].dcenter,
            (tz_ec_top_x, top_facet_y - tz_ec_top.dysize - 1.5),
        )
        gf.routing.route_single(
            c,
            port1=tz_ec_top.ports["o2"],
            port2=tz_output_port,
            cross_section="xs_rwg2000",
            bend=routing_bend,
            radius=routing_roc,
            straight="straight_rwg2000",
        )

    # PM Edge Coupler and routing
    if "o1" in pm_ref.ports:
        pm_input_port = pm_ref.ports["o1"]
        pm_ec_bottom = c << components.tilted_inverse_taper_AQ(taper_length=200, input_ext=12)
        pm_ec_bottom.drotate(0)
        left_facet_x = chip_layout.xmin + input_ext
        pm_ec_bottom.dmove(
            pm_ec_bottom.ports["o2"].dcenter,
            (left_facet_x + pm_ec_bottom.dxmax, pm_input_port.center[1]),
        )
        gf.routing.route_single(
            c,
            port1=pm_ec_bottom.ports["o2"],
            port2=pm_input_port,
            cross_section="xs_rwg2000",
            bend=routing_bend,
            radius=routing_roc,
            straight="straight_rwg2000"
        )

    #AM Edge Coupler and routing
    if "o1" in am_ref.ports:
        am_input_port = am_ref.ports["o1"]
        am_ec_bottom = c << components.tilted_inverse_taper_AQ(taper_length=200, input_ext=12)
        am_ec_bottom.drotate(0)
        left_facet_x = chip_layout.xmin + input_ext
        am_ec_bottom.dmove(
            am_ec_bottom.ports["o2"].dcenter,
            (left_facet_x + am_ec_bottom.dxmax, am_input_port.center[1]),
        )
        gf.routing.route_single(
            c,
            port1=am_ec_bottom.ports["o2"],
            port2=am_input_port,
            cross_section="xs_rwg2000",
            bend=routing_bend,
            radius=routing_roc,
            straight="straight_rwg2000"
        )

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
