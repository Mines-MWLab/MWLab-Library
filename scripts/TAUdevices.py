# Devices by TAU

import os
import sys
from pathlib import Path

import gdsfactory as gf
import gdsfactory.path as gp
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec, ComponentSpec
import numpy as np
import math

if __package__:
    from . import components as orc_components
else:  # pragma: no cover - convenience for script execution
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = Path(__file__).resolve().parent
    for _path in (scripts_dir, repo_root):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

    try:
        import components as orc_components  # type: ignore
    except ImportError as _exc:
        raise ImportError(
            "Unable to import 'components'. Ensure you run from the repository root or use 'python -m scripts.TAUdevices'."
        ) from _exc

from functools import partial
import matplotlib.pyplot as plt
from lnoi400.tech import LAYER, xs_uni_cpw
from lnoi400.cells import uni_cpw_straight, S_bend_vert, L_turn_bend, eo_phase_shifter
from gdsfactory.routing import route_single_sbend
from gdsfactory.routing import route_quad
from lnoi400.spline import (
    bend_S_spline,
    bend_S_spline_varying_width,
    spline_clamped_path,
)

#####################################################################################
# Authors: Ajwaad Quashef, ORC 2025


#####################################################################################
# Phase Modulated Active MLL Cavity = Edge Coupler + PM + Loop Mirror: Ajwaad Quashef
#####################################################################################
@gf.cell
def PM_MLL_cavity_AQ():

    #define constants


    #define subcomponents
    splitter = orc_components.custom_mmi_AQ()
    mirror = orc_components.loop_mirror_AQ(splitter='custom_mmi', cross_section='xs_rwg2000')

    input_ext = 10.0
    edge_coupler = orc_components.tilted_DL_inverse_taper_AQ(
                                input_ext=input_ext,
                                angle = 16.86,
                                )
    st_wg = orc_components.straight_rwg2000(
        length = 210
    )
    phase_modulator = orc_components.eo_phase_modulator_AQ(modulation_length=9230.0)


    def cavity():
        # push subcomponents to design
        cavity_component = gf.Component()
        # edge_coupler_ref = cavity_component << edge_coupler
        st_wg_ref = cavity_component << st_wg
        pm_ref = cavity_component << phase_modulator
        lm_ref = cavity_component << mirror
        
        # Position straight waveguide next to edge coupler
        # st_wg_ref.connect("o1", edge_coupler_ref.ports["o2"])
        
        # Position phase modulator next to straight waveguide
        pm_ref.connect("o1", st_wg_ref.ports["o2"])

        # Add y branch and loop mirror
        # yb_ref.connect("o1", pm_ref.ports["o2"])
        lm_ref.connect("o1", pm_ref.ports["o2"])

        # Add ports to the component
        cavity_component.add_port("o1", port=st_wg_ref.ports["o1"])

        #cavity_component.flatten()

        return cavity_component

    # create assembly
    c = gf.Component()
    cavity_ref = c << cavity()
    c.add_ports(cavity_ref.ports)

    return c


#############################################################################################
# Amplitude Modulated Active MLL Cavity = Edge Coupler + MZM + Loop Mirror: Ajwaad Quashef
#############################################################################################
@gf.cell
def AM_MLL_cavity_AQ():
    #define constants


    #define subcomponents
    mirror = orc_components.loop_mirror_AQ(splitter='custom_mmi_AQ', cross_section='xs_rwg2000')
    mzm = orc_components.mzm_custom_AQ(modulation_length=8540)

    input_ext = 10.0
    edge_coupler = orc_components.tilted_DL_inverse_taper_AQ(
                                input_ext=input_ext,
                                angle = 16.86,
                                )

    st_wg = orc_components.straight_rwg2000(
        length = 10
    )


    def cavity():
        # push subcomponents to design
        cavity_component = gf.Component()
        # edge_coupler_ref = cavity_component << edge_coupler
        st_wg_ref = cavity_component << st_wg
        mzm_ref = cavity_component << mzm
        lm_ref = cavity_component << mirror
        
        # Position straight waveguide next to edge coupler
        # st_wg_ref.connect("o1", edge_coupler_ref.ports["o2"])
        
        # Position MZM next to straight waveguide and connect loop mirror
        mzm_ref.connect("o1", st_wg_ref.ports["o2"])
        lm_ref.connect("o1", mzm_ref.ports["o2"])

        # Add ports to the component
        cavity_component.add_port("o1", port=st_wg_ref.ports["o1"])

        #cavity_component.flatten()

        return cavity_component

    # create assembly
    c = gf.Component()
    cavity_ref = c << cavity()
    c.add_ports(cavity_ref.ports)

    return c

##################################################################################### End: AQ


#####################################################################################
# Author: Redwan Islam, ORC 2025

########################################################################################################
# MZI-based Tunable Laser Components - custom_MZM, custom_MMI(placeholder), routing, eo_phase_modulator
#########################################################################################################

@gf.cell
def tunable_mzm_laser_Redwan():
    # 1. Create the main component that will hold the entire circuit
    c = gf.Component('MZM_Array')

    # 2. Define the settings for the MZM components
    # Common settings for all MZMs
    mzm_common_settings = {
        "modulation_length": 2000.0,
        "bias_tuning_section_length": 50,
        "lbend_tune_arm_reff": 150.0,
        "rf_central_conductor_width": 10.0,
        "rf_gap": 4.5,
        "with_heater": False,
    }

    # Specific 'length_imbalance' for each MZM variant
    mzm_variants = {
        "fine": {"length_imbalance": 5461},
        "medium": {"length_imbalance": 682.71},
        "coarse": {"length_imbalance": 54.6174},
    }

    # 3. Define desired global positions for each MMI (inputs)
    mmi_positions = {
        "fine": (0, 0),
        "medium": (-820, 650),
        "coarse": (-820, 1800),
    }

    # 4. Create and place the input MMIs at the specified positions
    mmi_component = orc_components.custom_mmi_AQ()
    main_input_mmi = c.add_ref(mmi_component)
    input_mmi_references = {}

    for name in mzm_variants.keys():
        mmi_ref = c.add_ref(mmi_component)
        mmi_ref.move(mmi_positions[name])
        input_mmi_references[name] = mmi_ref

    # 5. Create, place, and connect the MZM components + one output MMI per MZM
    output_mmi_references = {}

    for name, variant_settings in mzm_variants.items():
        # Merge common + variant settings
        current_mzm_settings = {**mzm_common_settings, **variant_settings}

        # Create MZM
        mzm_component = orc_components.custom_mzm(**current_mzm_settings)
        mzm_ref = c.add_ref(mzm_component)

        # Connect input ports
        mzm_ref.connect('in_top', input_mmi_references[name].ports['o2'])
        mzm_ref.connect('in_bot', input_mmi_references[name].ports['o3'])

        # --- Add a single output MMI per MZM and connect its o2/o3 ---
        out_mmi = c.add_ref(mmi_component)

        # Connect the output MMI's o2 to the MZM out_top and o3 to out_bot
        out_mmi.connect('o2', mzm_ref.ports['out_bot'])
        # store reference to use it later
        output_mmi_references[name] = out_mmi

    # 6. Define the start and end ports for the new route
    ##output single ports of the mmis
    coarse_output_mmi = output_mmi_references['coarse'].ports['o1']
    medium_output_mmi = output_mmi_references['medium'].ports['o1']
    fine_output_mmi = output_mmi_references['fine'].ports['o1']

    ##input single ports of the mmis
    coarse_input_mmi = input_mmi_references['coarse'].ports['o1']
    medium_input_mmi = input_mmi_references['medium'].ports['o1']
    fine_input_mmi = input_mmi_references['fine'].ports['o1']

    ##main input mmi connection
    # main_input_mmi.connect('o3', input_mmi_references['coarse'].ports['o1'])
    main_input_mmi.move((-1200, 2150))
    main_input_mmi_bot_right_port = main_input_mmi.ports['o3']
    main_input_mmi_top_right_port = main_input_mmi.ports['o2']

    # Create and move the phase shifter first
    phase_shifter = orc_components.single_custom_ps(modulation_length=2000)
    ps_ref = c.add_ref(phase_shifter)
    ps_ref.movey(2500)
    ps_ref.movex(1000)

    ps_ref_right_port = ps_ref.ports['top_o2']
    ps_ref_left_port = ps_ref.ports['top_o1']

    waypoint = [(3550, 3250)]

    # 7. Generate the waveguide route
    ## routing between coarse and medium
    route1 = gf.routing.route_single(c,
                                     port1=coarse_output_mmi,
                                     port2=medium_output_mmi,
                                     cross_section='xs_rwg2000',
                                     bend='L_turn_bend',
                                     radius=100,
                                     straight='straight_rwg2000'
                                     )

    ## routing between medium and fine
    route2 = gf.routing.route_single(c,
                                     port1=medium_input_mmi,
                                     port2=fine_input_mmi,
                                     cross_section='xs_rwg2000',
                                     bend='L_turn_bend',
                                     radius=100,
                                     straight='straight_rwg2000'
                                     )

    ## routing between phase shifter and fine
    route3 = gf.routing.route_single(c,
                                     port1=ps_ref_right_port,
                                     port2=fine_output_mmi,
                                     cross_section='xs_rwg2000',
                                     bend='L_turn_bend',
                                     radius=100,
                                     straight='straight_rwg2000',
                                     steps=[
                                         {'x': 3000},  # 1. Go straight until x=3000
                                         {'y': 3150},  # 2. Turn and go straight until y=3150
                                         {'x': 4000},  # 3. Turn and go straight until x=4000
                                         {'y': 0},  # 4. Turn and go straight until y=0
                                     ],
                                     )

    ## routing between main input mmi and phase shifter
    route4 = gf.routing.route_single(c,
                                     port1=main_input_mmi_top_right_port,
                                     port2=ps_ref_left_port,
                                     cross_section='xs_rwg2000',
                                     bend='L_turn_bend',
                                     radius=100,
                                     straight='straight_rwg2000',
                                     )

    ## routing between main input mmi and coarse
    route5 = gf.routing.route_single(c,
                                     port1=main_input_mmi_bot_right_port,
                                     port2=coarse_input_mmi,
                                     cross_section='xs_rwg2000',
                                     bend='L_turn_bend',
                                     radius=100,
                                     straight='straight_rwg2000',
                                     )

    # Expose the leftmost waveguide (main input MMI) as the device port
    c.add_port("o1", port=main_input_mmi.ports["o1"])

    return c

@gf.cell

def soliton_ring_Redwan(coupling_gap: float = 2.0):
    """
    Creates the soliton ring resonator by calling the concentric rings component.

    Args:
        coupling_gap_bus: The gap between the bus waveguide and the outer ring.
    """
    # Call the component from the 'components' file, passing the specified gap.
    # The other parameters will use their default values.
    c = orc_components.concentric_rings_with_bus(
        coupling_gap_bus=coupling_gap
    )
    return c

##################################################################################### End: Redwan Islam


#####################################################################################
# Ring Vortex Beam Emitter with Notches: Andrea Caruso (adapted for LNOI400 PDK)
#####################################################################################
@gf.cell
def ring_vortex_beam_emitter_AC(
    q: int = 338,
    R_ring: float = 50.0,
    W_wg: float = 0.8,
    W_gap: float = 0.3,
    W_notch: float = 0.25,
    notch_type: str = "S",
    resolution: float = 0.1,
    show_ports: bool = False,
    layer: tuple[int, int] | object = LAYER.LN_RIDGE,
    cross_section: CrossSectionSpec = "xs_rwg800",
) -> gf.Component:
    """Microring resonator with rectangular or circular notches.

    Args:
        q: Number of notches distributed along the ring perimeter.
        R_ring: Ring radius measured on the waveguide centerline (µm).
        W_wg: Bus and ring waveguide width (µm). Must match the chosen cross section.
        W_gap: Coupling gap between ring and bus waveguide (µm).
        W_notch: Width/diameter of each notch (µm).
        notch_type: "S" for rectangular slot, "C_in" for circular intrusion.
        resolution: Angular resolution used when discretizing the ring (deg per segment).
        show_ports: If True, draw port markers on the resulting component.
        layer: Target GDS layer/datatype for auxiliary geometries (defaults to LN_RIDGE).
        cross_section: PDK cross section for the ring and bus waveguides.
    """

    if q <= 0:
        raise ValueError("q must be a positive integer")
    if W_notch <= 0:
        raise ValueError("W_notch must be > 0")

    layer_tuple: tuple[int, int]
    if isinstance(layer, tuple):
        layer_tuple = layer
    elif hasattr(layer, "layer") and hasattr(layer, "datatype"):
        layer_tuple = (layer.layer, layer.datatype)
    elif isinstance(layer, str) and hasattr(LAYER, layer):
        layer_enum = getattr(LAYER, layer)
        layer_tuple = (layer_enum.layer, layer_enum.datatype)
    else:
        raise ValueError("layer must be provided as a tuple or known key in LAYER")

    xs = gf.get_cross_section(cross_section)
    xs_width = xs.width
    if abs(xs_width - W_wg) > 1e-3:
        raise ValueError(
            f"Cross-section width {xs_width} µm does not match W_wg={W_wg} µm. "
            "Add the appropriate cross-section to the PDK or adjust W_wg."
        )

    component = gf.Component(name="ring_vortex_beam_emitter_AC")

    # Build the microring by extruding the cross section along a circular path
    npoints = max(16, int(math.ceil(360.0 / resolution))) if resolution > 0 else 3600
    ring_path = gp.arc(radius=R_ring, angle=360, npoints=npoints)
    ring = gp.extrude(ring_path, cross_section=xs)
    ring_ref = component << ring
    ring_ref.center = (0.0, 0.0)

    rad_step = 2 * math.pi / q
    grid = 0.001  # µm grid assumed by layout

    def snap(value: float) -> float:
        return round(value / grid) * grid

    W_margin = snap(0.25 * W_notch)
    notch_width = snap(W_notch)
    notch_length = snap(W_notch + W_margin)

    notch_rect = gf.components.rectangle(
        size=(notch_length, notch_width),
        layer=layer_tuple,
        centered=True,
    )
    circle_layer = (layer_tuple[0], layer_tuple[1] + 1)
    notch_circle = gf.components.circle(
        radius=notch_width / 2,
        angle_resolution=2.5,
        layer=circle_layer,
    )

    for i in range(q):
        angle_rad = i * rad_step
        angle_deg = math.degrees(angle_rad)

        if notch_type == "S":
            notch_ref = component.add_ref(notch_rect)
            notch_ref.drotate(angle_deg)
            R_i = R_ring - notch_width - W_wg / 2
            notch_ref.dmove((snap(R_i * math.cos(angle_rad)), snap(R_i * math.sin(angle_rad))))
        elif notch_type == "C_in":
            notch_ref = component.add_ref(notch_circle)
            notch_ref.drotate(angle_deg)
            R_i = R_ring - notch_width / 2 - snap(0.3) - W_wg / 2
            notch_ref.dmove((snap(R_i * math.cos(angle_rad)), snap(R_i * math.sin(angle_rad))))
        else:
            raise ValueError("Unsupported notch_type. Use 'S' or 'C_in'.")

    bus_length = 2 * R_ring + W_wg
    bus = gf.components.straight(length=bus_length, cross_section=xs)
    bus_ref = component.add_ref(bus)
    bus_ref.dmovex(-bus_length / 2)
    bus_ref.dmovey(snap(-(R_ring + W_gap + W_wg)))

    o1_port = bus_ref.ports["o2"]
    o2_port = bus_ref.ports["o1"]

    component.flatten()

    component.add_port(
        "o1",
        center=(snap(o1_port.center[0]), snap(o1_port.center[1])),
        width=o1_port.width,
        orientation=o1_port.orientation,
        layer=o1_port.layer,
    )
    component.add_port(
        "o2",
        center=(snap(o2_port.center[0]), snap(o2_port.center[1])),
        width=o2_port.width,
        orientation=o2_port.orientation,
        layer=o2_port.layer,
    )

    if show_ports:
        component.draw_ports()

    return component


#####################################################################################
# Archimedean Spiral Vortex Beam Emitter: Andrea Caruso (adapted for LNOI400 PDK)
#####################################################################################
@gf.cell
def spiral_vortex_beam_emitter_equal_arc_spacing_AC(
    q: int = 331,
    W_spiral: float = 0.8,
    W_notch: float = 0.25,
    notch_type: str = "S",
    variable_pillar_dist: bool = False,
    loops_spiral: int = 1,
    loops_tail: int = 1,
    R_sep: float = 3.0,
    R_min: float = 48.0,
    direction: str = "R",
    resolution: int = 1000,
    layer: tuple[int, int] | object = LAYER.LN_RIDGE,
    show_ports: bool = False,
    cross_section: CrossSectionSpec = "xs_rwg800",
) -> gf.Component:
    """Archimedean spiral vortex beam emitter compatible with the LNOI400 PDK."""

    if q <= 0:
        raise ValueError("q must be a positive integer")
    if loops_spiral <= 0 or loops_tail < 0:
        raise ValueError("loops_spiral must be > 0 and loops_tail >= 0")
    if R_sep <= 0 or R_min <= 0:
        raise ValueError("R_sep and R_min must be positive")
    if resolution < 4:
        raise ValueError("resolution must be >= 4 to resolve the spiral path")

    layer_tuple: tuple[int, int]
    if isinstance(layer, tuple):
        layer_tuple = layer
    elif hasattr(layer, "layer") and hasattr(layer, "datatype"):
        layer_tuple = (layer.layer, layer.datatype)
    elif isinstance(layer, str) and hasattr(LAYER, layer):
        layer_enum = getattr(LAYER, layer)
        layer_tuple = (layer_enum.layer, layer_enum.datatype)
    else:
        raise ValueError("layer must be provided as a tuple or known key in LAYER")

    xs_main = gf.get_cross_section(cross_section, width=W_spiral)
    if abs(xs_main.width - W_spiral) > 1e-3:
        raise ValueError(
            f"Cross-section width {xs_main.width} µm does not match W_spiral={W_spiral} µm."
        )

    grid = 0.001

    def snap(value: float) -> float:
        return round(value / grid) * grid

    def equal_arc_spiral(a: float, r0: float, theta_max: float, samples: int) -> tuple[np.ndarray, np.ndarray, float]:
        samples = max(samples, int(q * 4))
        theta_samples = np.linspace(0.0, theta_max, samples)
        r_samples = a * theta_samples + r0
        integrand = np.sqrt(a**2 + r_samples**2)
        cumulative = np.concatenate(
            (
                [0.0],
                np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(theta_samples)),
            )
        )
        total_length = float(cumulative[-1])
        targets = np.linspace(0.0, total_length, q + 1)
        theta_values = np.interp(targets, cumulative, theta_samples)
        radii_values = a * theta_values + r0
        return theta_values, radii_values, total_length

    def taper_spiral_waveguide(
        separation: float,
        width_in: float,
        width_tip: float,
        number_of_loops: int,
        min_bend_radius: float,
        npoints: int,
        taper: str = "linear",
    ) -> gf.Component:
        if min_bend_radius <= 0:
            raise ValueError("min_bend_radius must be positive for the taper spiral")
        xs_in = gf.get_cross_section(cross_section, width=width_in)
        xs_tip = gf.get_cross_section(cross_section, width=width_tip)
        xs_transition = gp.transition(cross_section1=xs_tip, cross_section2=xs_in, width_type=taper)
        path = gp.spiral_archimedean(
            min_bend_radius=min_bend_radius,
            separation=separation / 2,
            number_of_loops=number_of_loops,
            npoints=npoints,
        )
        path.start_angle = 0
        path.end_angle = 0
        return gp.extrude_transition(path, xs_transition)

    L_in = R_min + loops_spiral * R_sep - W_spiral / 2
    if L_in <= 0:
        raise ValueError("Computed input straight length is non-positive; adjust R_min or loops_spiral")

    component = gf.Component(name="spiral_vortex_beam_emitter_equal_arc_spacing_AC")

    spiral_ref = component.add_ref(
        taper_spiral_waveguide(
            separation=R_sep,
            width_in=W_spiral,
            width_tip=W_spiral,
            number_of_loops=loops_spiral,
            min_bend_radius=R_min,
            npoints=resolution,
        )
    )
    spiral_ref.drotate(270)

    a = -R_sep / (2 * np.pi)
    r0 = R_min + loops_spiral * R_sep
    theta_max = loops_spiral * 2 * np.pi
    theta_values, radii_values, _ = equal_arc_spiral(a, r0, theta_max, resolution)

    W_margin = snap(0.25 * W_notch)
    notch_width = snap(W_notch)
    notch_length = snap(W_notch + W_margin)

    notch_rect = gf.components.rectangle(size=(notch_length, notch_width), layer=layer_tuple, centered=True)
    circle_layer = (layer_tuple[0], layer_tuple[1] + 1)
    notch_circle = gf.components.circle(radius=notch_width / 2, angle_resolution=2.5, layer=circle_layer)

    gap_i = 0.4

    for theta, radius_at_spot in zip(theta_values, radii_values):
        angle_deg = math.degrees(theta)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        if notch_type == "S":
            notch_ref = component << notch_rect
            notch_ref.drotate(angle_deg)
            radial_offset = radius_at_spot - notch_width - W_spiral / 2
            notch_ref.dmovex(snap(radial_offset * cos_t))
            notch_ref.dmovey(snap(radial_offset * sin_t))
        elif notch_type == "C_in":
            notch_ref = component << notch_circle
            notch_ref.drotate(angle_deg)
            radial_offset = radius_at_spot - notch_width / 2 - gap_i - W_spiral / 2
            notch_ref.dmovex(snap(radial_offset * cos_t))
            notch_ref.dmovey(snap(radial_offset * sin_t))
            if variable_pillar_dist:
                gap_i = gap_i - 0.1 * 1 / max(q, 1)
        elif notch_type == "C_out":
            notch_ref = component << notch_circle
            notch_ref.drotate(angle_deg)
            radial_offset = radius_at_spot + notch_width / 2 + gap_i + W_spiral / 2
            notch_ref.dmovex(snap(radial_offset * cos_t))
            notch_ref.dmovey(snap(radial_offset * sin_t))
            if variable_pillar_dist:
                gap_i = gap_i - 0.1 * 1 / max(q, 1)
        else:
            raise ValueError("Unsupported notch_type. Use 'S', 'C_in', or 'C_out'.")

    if loops_tail:
        tail_ref = component.add_ref(
            taper_spiral_waveguide(
                separation=5 * R_sep,
                width_in=W_spiral,
                width_tip=0.25,
                number_of_loops=loops_tail,
                min_bend_radius=R_min - loops_tail * 5 * R_sep,
                npoints=resolution,
            )
        )
        tail_ref.drotate(270)

    component.rotate(360 / q)

    R_max = R_min + loops_spiral * R_sep

    arc_path = gp.arc(radius=R_max, angle=360 / q, npoints=max(16, int(resolution / max(q, 1))))
    arc_component = gp.extrude(arc_path, cross_section=xs_main)
    arc_ref = component.add_ref(arc_component)
    arc_ref.drotate(90)
    arc_ref.dmovex(R_max)

    straight_ref = component.add_ref(gf.components.straight(length=L_in, cross_section=xs_main))
    straight_ref.drotate(90)
    straight_ref.dmovex(R_max)
    straight_ref.dmovey(-L_in)

    component.flatten()

    handedness = direction.upper()
    if handedness == "L":
        component.mirror_x()
        port_center = (snap(-R_max), snap(-L_in))
    elif handedness == "R":
        port_center = (snap(R_max), snap(-L_in))
    else:
        raise ValueError("direction must be either 'L' or 'R'")

    component.add_port(
        "o1",
        center=port_center,
        width=W_spiral,
        orientation=270,
        layer=layer_tuple,
    )

    if show_ports:
        component.draw_ports()

    return component


@gf.cell
def NanoPh_eo_phase_shifter(
    modulation_length: float = 7500.0,
    taper_length: float = 100.0,
    rib_core_width_modulator: float = 2.5,
    rf_central_conductor_width: float = 10.0,
    rf_gap: float = 4.0,
    rf_ground_planes_width: float = 180.0,
    cpw_cell: ComponentSpec = uni_cpw_straight,
    draw_cpw: bool = True,
) -> gf.Component:
    c = gf.Component("NanoPh_eo_phase_shifter")

    ps = eo_phase_shifter(
        modulation_length=modulation_length,
        taper_length=taper_length,
        rib_core_width_modulator=rib_core_width_modulator,
        rf_central_conductor_width=rf_central_conductor_width,
        rf_gap=rf_gap,
        rf_ground_planes_width=rf_ground_planes_width,
        cpw_cell=cpw_cell,
        draw_cpw=draw_cpw,
    )

    ps_ref = c << ps
    c.add_ports(ps_ref.ports)
    c.info["modulation_length"] = modulation_length
    c.info["taper_length"] = taper_length
    c.info["rib_core_width_modulator"] = rib_core_width_modulator
    c.info["rf_central_conductor_width"] = rf_central_conductor_width
    c.info["rf_gap"] = rf_gap
    c.info["rf_ground_planes_width"] = rf_ground_planes_width

    return c
