import gdsfactory as gf
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec
from gdsfactory.typings import ComponentSpec
from lnoi400.tech import LAYER, xs_uni_cpw
from lnoi400.spline import bend_S_spline_varying_width
import numpy as np
from gdsfactory.routing import route_single
import pathlib
from gdsfactory.path import arc
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
from lnoi400.cells import uni_cpw_straight, S_bend_vert, L_turn_bend
from gdsfactory.routing import route_single_sbend
from gdsfactory.routing import route_quad
from lnoi400.spline import (
    bend_S_spline,
    bend_S_spline_varying_width,
    spline_clamped_path,
)

#####################
# Asymmetric directional coupler
#####################
gf.clear_cache()
@gf.cell
def asymmetric_directional_coupler(
    io_wg_sep: float = 30.6,
    sbend_length: float = 58,
    central_straight_length: float = 16.92,
    coupl_wg_sep: float = 1.5,
    coup_wg_width: float = 1.0,
    taper_npoints: int = 201,
    cross_section_io: CrossSectionSpec = "xs_rwg1000",
) -> gf.Component:
    """Returns a 50-50 directional coupler. Default parameters give a 50/50 splitting at 1550 nm.

    Args:
        io_wg_sep: Separation of the two straights at the input/output, top-to-top.
        sbend_length: length of the s-bend part.
        central_straight_length: length of the coupling region.
        coupl_wg_sep: Distance between two waveguides in the coupling region (side to side).
        cross_section_io: cross section spec at the i/o (must be defined in tech.py).
        coup_wg_width: waveguide width at the coupling section.
    """

    s0 = gf.Section(
        width=coup_wg_width,
        offset=0,
        layer="LN_RIDGE",
        name="_default",
        port_names=("o1", "o2"),
    )
    s1 = gf.Section(
        width=10.0, 
        offset=0, 
        layer="LN_SLAB", 
        name="slab", 
        simplify=0.03
    )
    
    cross_section_coupling = gf.CrossSection(sections=[s0, s1])
    cross_section_io = gf.get_cross_section(cross_section_io)

    s_height = (
        io_wg_sep - coupl_wg_sep - coup_wg_width
    ) / 2  # take into account the width of the waveguide
    size = (sbend_length, s_height)

    # s-bend settings
    settings_s_bend = {
        "size": size,
        "cross_section1": cross_section_coupling,
        "cross_section2": cross_section_io,
        "npoints": 201,
    }
    dc = gf.Component()

    # top right branch
    
    c_tr = dc << lnoi400.cells.bend_S_spline_varying_width(
        size= (sbend_length, 0), cross_section1="xs_rwg1000", cross_section2="xs_rwg3000", npoints=taper_npoints
    )
    c_tr.dmove(
        c_tr.ports["o1"].dcenter, (central_straight_length / 2, 
                                   0.5 * (coupl_wg_sep + coup_wg_width)),
    )
    
    # bottom right branch
    c_br = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_br.dmirror_y()
    c_br.dmove(
        c_br.ports["o1"].dcenter,
        (central_straight_length / 2, -0.5 * (coupl_wg_sep + coup_wg_width)),
    )
    gf.components.taper
    # central waveguide
    straight_center_up = dc << gf.components.straight(
        length=central_straight_length, cross_section=cross_section_io
    )
    straight_center_up.connect("o2", c_tr.ports["o1"])

    straight_center_down = dc << gf.components.straight(
        length=central_straight_length, cross_section=cross_section_coupling
    )
    straight_center_down.connect("o2", c_br.ports["o1"])

    # top left branch
    c_tl = dc << lnoi400.cells.bend_S_spline_varying_width(
        size= (sbend_length, 0), cross_section1="xs_rwg3000", cross_section2="xs_rwg1000", npoints=taper_npoints
    )
    c_tl.dmove(
        c_tl.ports["o1"].dcenter, (- sbend_length - central_straight_length / 2, 
                                   0.5 * (coupl_wg_sep + coup_wg_width)),
    )

    # bottom left branch
    c_bl = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_bl.dmirror_x()
    c_bl.dmirror_y()
    c_bl.dmove(c_bl.ports["o1"].dcenter, straight_center_down.ports["o1"].dcenter)

    # Expose the ports
    exposed_ports = [
        ("o1", c_bl.ports["o2"]),
        ("o2", c_tl.ports["o2"]),
        ("o3", c_tr.ports["o2"]),
        ("o4", c_br.ports["o2"]),
    ]

    [dc.add_port(name=name, port=port) for name, port in exposed_ports]
    return dc


@gf.cell
def asymmetric_directional_coupler_racetrack(
    io_wg_sep: float = 30.6,
    sbend_length: float = 58,
    central_straight_length: float = 16.92,
    coupl_wg_sep: float = 1.5,
    coup_wg_width: float = 1.0,
    taper_npoints: int = 201,
    cross_section_io: CrossSectionSpec = "xs_rwg1000",
    cross_section_bus: CrossSectionSpec = "xs_rwg1000",
) -> gf.Component:
    """Returns a 50-50 directional coupler. Default parameters give a 50/50 splitting at 1550 nm.

    Args:
        io_wg_sep: Separation of the two straights at the input/output, top-to-top.
        sbend_length: length of the s-bend part.
        central_straight_length: length of the coupling region.
        coupl_wg_sep: Distance between two waveguides in the coupling region (side to side).
        cross_section_io: cross section spec at the i/o (must be defined in tech.py).
        coup_wg_width: waveguide width at the coupling section.
    """

    s0 = gf.Section(
        width=coup_wg_width,
        offset=0,
        layer="LN_RIDGE",
        name="_default",
        port_names=("o1", "o2"),
    )
    s1 = gf.Section(
        width=9.0+coup_wg_width, 
        offset=0, 
        layer="LN_SLAB", 
        name="slab", 
        simplify=0.03
    )
    
    cross_section_coupling = gf.CrossSection(sections=[s0, s1])
    cross_section_io = gf.get_cross_section(cross_section_io)

    bus_wg_width = gf.get_cross_section(cross_section_bus).width
    s_height = (
        io_wg_sep - coupl_wg_sep - coup_wg_width
    ) / 2  # take into account the width of the waveguide
    size = (sbend_length, s_height*0)

    # s-bend settings
    settings_s_bend = {
        "size": size,
        "cross_section1": cross_section_coupling,
        "cross_section2": cross_section_io,
        "npoints": 201,
    }
    dc = gf.Component()

    # top right branch
    
    c_tr = dc << lnoi400.cells.bend_S_spline_varying_width(
        size= (sbend_length, s_height), cross_section1=cross_section_bus, cross_section2=cross_section_bus, npoints=taper_npoints
    )
    c_tr.dmove(
        c_tr.ports["o1"].dcenter, (central_straight_length / 2, 
                                   0.5 * (coupl_wg_sep + bus_wg_width)),
    )
    
    # bottom right branch
    c_br = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_br.dmirror_y()
    c_br.dmove(
        c_br.ports["o1"].dcenter,
        (central_straight_length / 2, -0.5 * (coupl_wg_sep + coup_wg_width)),
    )
    gf.components.taper
    # central waveguide
    straight_center_up = dc << gf.components.straight(
        length=central_straight_length, cross_section=cross_section_bus
    )
    straight_center_up.connect("o2", c_tr.ports["o1"])

    straight_center_down = dc << gf.components.straight(
        length=central_straight_length, cross_section=cross_section_coupling
    )
    straight_center_down.connect("o2", c_br.ports["o1"])

    # top left branch
    c_tl = dc << lnoi400.cells.bend_S_spline_varying_width(
        size= (sbend_length, -s_height), cross_section1=cross_section_bus, cross_section2=cross_section_bus, npoints=taper_npoints
    )
    c_tl.dmove(
        c_tl.ports["o1"].dcenter, (- sbend_length - central_straight_length / 2, 
                                   0.5 * (coupl_wg_sep + bus_wg_width) + s_height),
    )

    # bottom left branch
    c_bl = dc << bend_S_spline_varying_width(**settings_s_bend)
    c_bl.dmirror_x()
    c_bl.dmirror_y()
    c_bl.dmove(c_bl.ports["o1"].dcenter, straight_center_down.ports["o1"].dcenter)

    # Expose the ports
    exposed_ports = [
        ("o4", c_bl.ports["o2"]),
        ("o1", c_tl.ports["o1"]),
        ("o2", c_tr.ports["o2"]),
        ("o3", c_br.ports["o2"]),
    ]

    [dc.add_port(name=name, port=port) for name, port in exposed_ports]
    return dc

@gf.cell
def U_bend_racetrack_varang(
    angle: float = 180.0,
    v_offset: float = 90.0,
    p: float = 1.0,
    with_arc_floorplan: bool = True,
    cross_section: CrossSectionSpec = "xs_rwg3000",
    **kwargs,
) -> gf.Component:
    """A U-bend with fixed cross-section and dimensions, suitable for building a low-loss racetrack resonator."""
    
    radius = 0.5 * v_offset

    npoints = int(np.round(600 * radius / 90.0))
    #angle = 180.0

    return gf.components.bend_euler(
        radius=radius,
        angle=angle,
        p=p,
        with_arc_floorplan=with_arc_floorplan,
        npoints=npoints,
        cross_section=cross_section,
        **kwargs,
    )


#@gf.cell
@gf.cell(check_instances=False)
def pulley_coupler(
    ubend_diameter: float = 200.0,
    pulley_angle: float = 45.0,
    pulley_diameter: float = 300.0,
    pulley_gap: float = 0.6,
    cross_section_ubend: CrossSectionSpec = "xs_rwg3000",
    cross_section_pulley: CrossSectionSpec = "xs_rwg1000",
) -> gf.Component:
    """A U-bend with pulley coupler."""

    c1 = U_bend_racetrack_varang(angle=pulley_angle,v_offset=pulley_diameter, p=1.0, with_arc_floorplan=True, cross_section=cross_section_pulley)
    c2 = U_bend_racetrack_varang(angle=pulley_angle/2.0,v_offset=pulley_diameter, p=1.0, with_arc_floorplan=True, cross_section=cross_section_pulley)
    c3 = U_bend_racetrack_varang(angle=pulley_angle/2.0,v_offset=pulley_diameter, p=1.0, with_arc_floorplan=True, cross_section=cross_section_pulley)
    c4 = U_bend_racetrack_varang(angle=180.0,v_offset=ubend_diameter, p=1.0, with_arc_floorplan=True, cross_section=cross_section_ubend)
    c = gf.Component()
    wg1 = c << c1
    wg2 = c << c2
    wg3 = c << c3
    wg2.connect("o1",wg1.ports["o2"], mirror=True)
    #wg2.connect("o2",wg1.ports["o2"])
    wg3.connect("o1", wg1.ports["o1"])

    exposed_ports = [
        ("o1", wg2.ports["o2"]),
        ("o2", wg3.ports["o2"]),
    ]
    [c.add_port(name=name, port=port) for name, port in exposed_ports]

    c.rotate(90-pulley_angle+pulley_angle/2.0)


    # obtain width of bus
    c_gds = gf.read.import_gds(c.write_gds())
    c_gds.flatten(merge = True)
    c_gds.remove_layers([(3,0)])
    sizex_bus = c_gds.xsize

    # obtain width of racetrack
    c4_gds = gf.read.import_gds(c4.write_gds())
    c4_gds.flatten(merge = True)
    c4_gds.remove_layers([(3,0)])
    sizex_rt = c4_gds.xsize
    #print(sizex_bus, sizex_rt)

    size_buswg = gf.get_cross_section(cross_section_pulley).sections[0].width

    rt = c << c4

    rt.dmovey(0.5*rt.ports["o1"].center[1] + 0.5*rt.ports["o2"].center[1],
            0.5*wg3.ports["o2"].center[1]+ 0.5*wg2.ports["o2"].center[1])



    rt.dmovex(rt.ports["o1"].center[0],
            wg3.ports["o2"].center[0]+sizex_bus-0.5*size_buswg-size_buswg-sizex_rt-pulley_gap,
            )

    [c.add_port(name=name, port=port) for name, port in [ ("o3", rt.ports["o1"]),("o4", rt.ports["o2"])]]
    return c


#####################
# Gold RF component of EOM, edited from PDK to allow Tcells to be a the end of the component (https://luxtelligence.github.io/lxt_pdk_gf/cells.html#lnoi400.cells.trail_cpw)
#####################
@gf.cell()
def trail_cpw_mpl(
    length: float = 1000.0,
    signal_width: float = 21,
    gap_width: float = 4,
    th: float = 1.5,
    tl: float = 44.7,
    tw: float = 7.0,
    tt: float = 1.5,
    tc: float = 5.0,
    ground_planes_width: float = 180.0,
    rounding_radius: float = 0.5,
    bondpad: ComponentSpec = "CPW_pad_linear",
    cross_section: CrossSectionSpec = xs_uni_cpw,
) -> gf.Component:
    """A CPW transmission line with periodic T-rails on all electrodes."""

    # num_cells = np.floor(length / (tl + tc))
    # makes room for Tcells
    num_cells = np.round((length + tc) / (tl + tc))
    gap_width_corrected = gap_width + 2 * th + 2 * tt  # total gap width with T-rails

    # redefine cross section to include T-rails
    xs_cpw_trail = partial(
        cross_section,
        central_conductor_width=signal_width,
        gap=gap_width_corrected,
        ground_planes_width=ground_planes_width,
    )

    cpw = gf.Component()
    bp = gf.get_component(bondpad, cross_section=xs_cpw_trail)
    strght = cpw << gf.components.straight(length=length, cross_section=xs_cpw_trail)
    bp1 = cpw << bp
    bp2 = cpw << bp
    bp1.connect("e2", strght.ports["e1"])
    bp2.dmirror()
    bp2.connect("e2", strght.ports["e2"])
    cpw.add_ports(strght.ports)

    cpw.add_port(
        name="bp1",
        port=bp1.ports["e1"],
    )
    cpw.add_port(
        name="bp2",
        port=bp2.ports["e1"],
    )

    # Initiate T-rail polygon element. Create a bit more to ensure round corners close to electrodes
    trailpol = gf.kdb.DPolygon(
        [
            (tl, signal_width / 2),
            (tl, signal_width / 2 - tt),
            (0, signal_width / 2 - tt),
            (0, signal_width / 2),
            (tl / 2 - tw / 2, signal_width / 2),
            (tl / 2 - tw / 2, signal_width / 2 + th),
            (0, signal_width / 2 + th),
            (0, signal_width / 2 + th + tt),
            (tl, signal_width / 2 + th + tt),
            (tl, signal_width / 2 + th),
            (tl / 2 + tw / 2, signal_width / 2 + th),
            (tl / 2 + tw / 2, signal_width / 2),
        ]
    )

    # Create T-rail component
    trailcomp = gf.Component()
    _ = trailcomp.add_polygon(trailpol, layer=cross_section().layer)

    # Apply roc to the T-rail corners
    trailround = gf.Component()
    rinner = rounding_radius * 1000  # 	The circle radius of inner corners (in nm).
    router = rounding_radius * 1000  # 	The circle radius of outer corners (in nm).
    n = 30  # 	The number of points per full circle.

    for layer, polygons in trailcomp.get_polygons().items():
        for p in polygons:
            p_round = p.round_corners(rinner, router, n)
            trailround.add_polygon(p_round, layer=layer)

    # Create T-rail unit cell
    trail_uc = gf.Component()
    inc_t1 = trail_uc << trailround
    inc_t2 = trail_uc << trailround
    inc_t2.dmovey(gap_width_corrected - th)
    inc_t3 = trail_uc << trailround
    inc_t3.dmovey(-signal_width - th)
    inc_t4 = trail_uc << trailround
    inc_t4.dmovey(-signal_width - gap_width_corrected)

    # Place T-rails symmetrically w/r to bondpads

    dl_tr = 0.5 * (length - num_cells * tl - (num_cells - 1) * tc)

    [ref.dmovex(dl_tr) for ref in (inc_t1, inc_t2, inc_t3, inc_t4)]

    # Duplicate cell
    cpw.add_ref(
        trail_uc,
        columns=num_cells,
        rows=1,
        column_pitch=tl + tc,
    )

    cpw.flatten()

    return cpw


#####################################################################################
# Authors: Ajwaad Quashef, ORC 2025

@gf.cell
def _straight(
    length: float = 10.0,
    cross_section: CrossSectionSpec = "xs_rwg2000",
) -> gf.Component:
    return gf.components.straight(
        length=length,
        cross_section=cross_section,
    )

@gf.cell
def straight_rwg2000(length: float = 10.0, **kwargs) -> gf.Component:
    """Straight single mode (at 2.3 um) waveguide."""
    if "cross_section" not in kwargs:
        kwargs["cross_section"] = "xs_rwg2000"
    return _straight(
        length=length,
        **kwargs,
    )

@gf.cell
def custom_mmi_AQ(
    width_mmi: float = 10.0,
    length_mmi: float = 34.0,
    width_taper: float = 1.5,
    length_taper: float = 25.0,
    port_ratio: float = 1.0,
    cross_section: CrossSectionSpec = "xs_rwg2000",
) -> gf.Component:
    """Ybranch/MMI inverse optimized for broadband transmission at 2300 nm."""

    c = gf.Component()
    script_dir = pathlib.Path(__file__).parent.resolve()
    gds_file_path = script_dir / "utility_files" / "y_branch_3D.gds"

    y_branch_geom = gf.import_gds(gds_file_path)
    # y_branch_geom = gf.import_gds("S:/61501_Users/Ajwaad/LXT PDK/Layout/y_branch_3D.gds")
    y_branch_ref = c << y_branch_geom

    # Add tapered slab layer underneath the MMI
    # Get the bounding box of the imported geometry
    slab_width_start = 18  # Input side width
    slab_width_end = 28    # Output side width
    slab_length = y_branch_ref.xmax - y_branch_ref.xmin
    
    # Create tapered slab using polygon
    # Define the four corners of the trapezoid
    half_width_start = slab_width_start / 2
    half_width_end = slab_width_end / 2
    half_length = slab_length / 2
    
    # Create trapezoid points: (x, y) coordinates
    points = [
        (-half_length, -half_width_start),  # Bottom left
        (-half_length, half_width_start),   # Top left  
        (half_length, half_width_end),      # Top right
        (half_length, -half_width_end),     # Bottom right
    ]
    
    # Add polygon directly to component
    c.add_polygon(points, layer="LN_SLAB")
    
    # Get the center position for moving the entire component if needed
    slab_center_x = (y_branch_ref.xmin + y_branch_ref.xmax) / 2
    slab_center_y = (y_branch_ref.ymin + y_branch_ref.ymax) / 2
    # Note: The polygon is already positioned relative to the component origin

    c.add_port(
        name="o1",
        center=(-32, 0),  # (x, y) coordinate of the port center
        width=2,        # Width of the port in microns
        orientation=180,  # 180 degrees points West (left)
        layer="LN_RIDGE"
    )

    c.add_port(
        name="o2",
        center=(32, 5),
        width=2,
        orientation=0,    # 0 degrees points East (right)
        layer="LN_RIDGE"
    )

    c.add_port(
        name="o3",
        center=(32, -5),
        width=2,
        orientation=0,
        layer="LN_RIDGE"
    )

    return c


def loop_mirror_AQ(
    splitter: ComponentSpec = "custom_mmi_AQ",
    cross_section: CrossSectionSpec = "rwg2000",
) -> gf.Component:
    """Returns Sagnac loop_mirror.    """
    c = gf.Component()
    # splitter = gf.get_component(splitter)
    splitter = custom_mmi_AQ()
    cref = c.add_ref(splitter)
    sref1 = c << gf.components.bend_s_offset(offset=100.0, radius=100.0, cross_section=cross_section)
    sref2 = c << gf.components.bend_s_offset(offset=100.0, radius=100.0, cross_section=cross_section)
    sref2.dmirror_y()
    sref1.connect("o1", cref.ports["o2"])
    sref2.connect("o1", cref.ports["o3"])

    route_single(
        c,
        sref1.ports["o2"],
        sref2.ports["o2"],
        straight=gf.components.straight(cross_section=cross_section),
        bend=gf.components.bend_euler(radius=100.0, p=1, cross_section=cross_section), radius=100.0,
        cross_section=cross_section,
    )

    c.add_port(name="o1", port=cref.ports["o1"])
    return c


@gf.cell
def eo_phase_modulator_AQ(
    modulation_length: float = 4500.0,
    cross_section: CrossSectionSpec = "xs_rwg2000",
    # RF parameters for CPW line
    rf_central_conductor_width: float = 10.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    cpw_cell: ComponentSpec = lnoi400.cells.uni_cpw_straight,
    draw_cpw: bool = True,
) -> gf.Component:
    """
    Phase modulator with a constant rib waveguide width (no tapers), intended
    for the custom_mzm. The waveguide is located within the gap of a CPW
    transmission line.
    """
    ps = gf.Component()
    xs_modulator = gf.get_cross_section(cross_section)

    # The phase modulation section is a simple straight waveguide
    wg_phase_modulation = gf.components.straight(
        length=modulation_length, cross_section=xs_modulator
    )
    wg_ref = ps << wg_phase_modulation

    ps.add_port(name="o1", port=wg_ref.ports["o1"])
    ps.add_port(name="o2", port=wg_ref.ports["o2"])

    # Add the transmission line (CPW)
    if draw_cpw:
        xs_cpw = gf.partial(
            xs_uni_cpw,
            central_conductor_width=rf_central_conductor_width,
            ground_planes_width=rf_ground_planes_width,
            gap=rf_gap,
        )
        tl = ps << cpw_cell(
            length=modulation_length,
            cross_section=xs_cpw,
            gap_width=rf_gap,
            signal_width=rf_central_conductor_width,
            ground_planes_width=rf_ground_planes_width,
        )

        gap_eff = rf_gap + 2 * np.sum(
            [tl.cell.settings[key] for key in ("tt", "th") if key in tl.cell.settings]
        )

        tl.dmove(
            tl.ports["e1"].dcenter,
            (0.0, -0.5 * rf_central_conductor_width - 0.5 * gap_eff),
        )

        for name, port in [("e1", tl.ports["bp1"]), ("e2", tl.ports["bp2"])]:
            ps.add_port(name=name, port=port)

    ps.flatten()
    return ps


@gf.cell
def _mzm_interferometer_AQ(
    splitter: ComponentSpec = "custom_mmi_AQ",
    modulation_length: float = 4500.0,
    sbend_large_size: tuple[float, float] = (100.0, 50.0),
    sbend_small_size: tuple[float, float] = (100.0, -45.0),
    sbend_small_straight_extend: float = 5.0,
) -> gf.Component:
    interferometer = gf.Component()

    sbend_large = lnoi400.cells.S_bend_vert(
        v_offset=sbend_large_size[1], h_extent=sbend_large_size[0], dx_straight=5.0,
        cross_section="xs_rwg2000",
    )

    sbend_small = lnoi400.cells.S_bend_vert(
        v_offset=sbend_small_size[1],
        h_extent=sbend_small_size[0],
        dx_straight=sbend_small_straight_extend,
        cross_section="xs_rwg2000",
    )

    def branch_top():
        bt = gf.Component()
        sbend_1 = bt << sbend_large
        sbend_2 = bt << sbend_small
        pm = bt << eo_phase_modulator_AQ(
            modulation_length=modulation_length,
            draw_cpw=False,
        )
        sbend_3 = bt << sbend_small
        sbend_4 = bt << sbend_large  # Create separate mirrored large S-bend for return
        
        # Mirror the return path S-bends
        sbend_3.dmirror_x()  # Mirror the small S-bend
        sbend_4.dmirror_x()  # Mirror the large S-bend for return path
        
        # Connect components in sequence
        sbend_2.connect("o1", sbend_1.ports["o2"])
        pm.connect("o1", sbend_2.ports["o2"])
        sbend_3.connect("o1", pm.ports["o2"])
        sbend_4.connect("o1", sbend_3.ports["o2"])
        sbend_3.connect("o1", pm.ports["o2"])

        for name, port in [
            ("o1", sbend_1.ports["o1"]),
            ("o2", sbend_4.ports["o2"]),  # Output from sbend_4
            ("taper_start", pm.ports["o1"]),
        ]:
            bt.add_port(name=name, port=port)
        bt.flatten()

        return bt

    splt = custom_mmi_AQ()

    # Uniformly handle the cases of a 1x2 or 2x2 MMI
    if len(splt.ports) == 4:
        out_top = splt.ports["o3"]
        out_bottom = splt.ports["o4"]
        combiner_in_top = splt.ports["o3"]
        combiner_in_bottom = splt.ports["o4"]
    elif len(splt.ports) == 3:
        out_top = splt.ports["o2"]
        out_bottom = splt.ports["o3"]
        combiner_in_top = splt.ports["o2"]
        combiner_in_bottom = splt.ports["o3"]
    else:
        raise ValueError(f"Splitter cell not supported.")

    # Place components
    splt_ref = interferometer << splt  # Input splitter
    combiner_ref = interferometer << splt  # Output combiner (same component)
    bt = interferometer << branch_top()
    bb = interferometer << branch_top()
    
    # Mirror bottom branch
    bb.dmirror_y()
    
    # Connect splitter to phase modulators
    bt.connect("o1", out_top)
    bb.connect("o1", out_bottom)
    
    # Mirror combiner to reverse direction
    combiner_ref.dmirror_x()
    
    # Connect branches directly to combiner (no intermediate routing needed)
    # For a mirrored 1x2 MMI, the input ports become o2 and o3
    if len(splt.ports) == 3:
        # For 1x2 MMI: connect branch outputs directly to combiner input ports
        combiner_ref.connect("o2", bt.ports["o2"])
        combiner_ref.connect("o3", bb.ports["o2"])
    elif len(splt.ports) == 4:
        # For 2x2 MMI: connect branch outputs directly to combiner input ports
        combiner_ref.connect("o3", bt.ports["o2"])
        combiner_ref.connect("o4", bb.ports["o2"])

    # Expose the ports
    exposed_ports = [
        ("o1", splt_ref.ports["o1"]),  # Input
        ("upper_taper_start", bt.ports["taper_start"]),
        ("o2", combiner_ref.ports["o1"]),  # Combined output
    ]

    for name, port in exposed_ports:
        interferometer.add_port(name=name, port=port)
    interferometer.flatten()

    return interferometer


@gf.cell
def mzm_custom_AQ(
    modulation_length: float = 4500.0,
    rf_pad_start_width: float = 80.0,
    rf_central_conductor_width: float = 10.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 190.0,
    cpw_cell: ComponentSpec = lnoi400.cells.uni_cpw_straight,
    **kwargs,
) -> gf.Component:
    """Balanced Mach-Zehnder modulator based on the Pockels effect with an applied RF field.
    The modulator works in a differential push-pull configuration driven by a single GSG line.
    Simplified version without bias tuning sections."""

    mzm = gf.Component()

    # Transmission line subcell
    xs_cpw = gf.partial(
        xs_uni_cpw,
        central_conductor_width=rf_central_conductor_width,
        ground_planes_width=rf_ground_planes_width,
        gap=rf_gap,
    )

    rf_line = mzm << cpw_cell(
        bondpad={
            "component": "CPW_pad_linear",
            "settings": {
                "start_width": rf_pad_start_width,
                "length_straight": rf_pad_length_straight,
                "length_tapered": rf_pad_length_tapered,
            },
        },
        length=modulation_length,
        signal_width=rf_central_conductor_width,
        cross_section=xs_cpw,
        ground_planes_width=rf_ground_planes_width,
        gap_width=rf_gap,
    )

    rf_line.dmove(rf_line.ports["e1"].dcenter, (0.0, 0.0))

    # Interferometer subcell
    splitter = custom_mmi_AQ()

    sbend_large_AR = 3.6

    gap_eff = rf_gap + 2 * np.sum(
        [
            rf_line.cell.settings[key]
            for key in ("tt", "th")
            if key in rf_line.cell.settings
        ]
    )

    GS_separation = rf_pad_start_width * gap_eff / rf_central_conductor_width

    sbend_large_v_offset = (
        0.5 * rf_pad_start_width
        + 0.5 * GS_separation
        - 0.5 * splitter.settings["port_ratio"] * splitter.settings["width_mmi"]
    )

    sbend_small_straight_length = rf_pad_length_straight * 0.5

    interferometer = (
        mzm
        << partial(
            _mzm_interferometer_AQ,
            modulation_length=modulation_length,
            sbend_large_size=(
                sbend_large_AR * sbend_large_v_offset,
                sbend_large_v_offset,
            ),
            sbend_small_size=(
                rf_pad_length_straight
                + rf_pad_length_tapered
                - 2 * sbend_small_straight_length,
                -0.5
                * (
                    rf_pad_start_width
                    - rf_central_conductor_width
                    + GS_separation
                    - gap_eff
                ),
            ),
            sbend_small_straight_extend=sbend_small_straight_length,
            **kwargs,
        )()
    )

    interferometer.dmove(
        interferometer.ports["upper_taper_start"].dcenter,
        (0.0, 0.5 * (rf_central_conductor_width + gap_eff)),
    )

    # Expose the ports - now with combiner
    exposed_ports = [
        ("e1", rf_line.ports["bp1"]),
        ("e2", rf_line.ports["bp2"]),
        ("o1", interferometer.ports["o1"]),  # Input
        ("o2", interferometer.ports["o2"]),  # Combined output
    ]

    [mzm.add_port(name=name, port=port) for name, port in exposed_ports]

    return mzm


@gf.cell
def linear_inverse_taper_AQ(
    cross_section_start: CrossSectionSpec = "xs_rwg750",
    cross_section_end: CrossSectionSpec = "xs_rwg2000",
    taper_length: float = 20.0,
    input_ext: float = 0.0,
) -> gf.Component:
    """Inverse rib width taper for edge coupler"""

    taper = gf.components.taper_cross_section(
        cross_section1=cross_section_start,
        cross_section2=cross_section_end,
        length=taper_length,
        linear=True,
    )

    if input_ext:
        straight_ext = gf.components.straight(
            cross_section=cross_section_start,
            length=input_ext,
        )

    inverse_taper = gf.Component()
    if input_ext:
        sref = inverse_taper << straight_ext
        sref.dmovex(-input_ext)
    itref = inverse_taper << taper

    # Define the input and output optical ports
    inverse_taper.add_port(
        port=sref.ports["o1"]
    ) if input_ext else inverse_taper.add_port(port=itref.ports["o1"])
    inverse_taper.add_port(port=itref.ports["o2"])

    inverse_taper.flatten()

    return inverse_taper

##################################################################################### End: AQ





#####################################################################################
# Author: Redwan Islam, ORC 2025


@gf.cell
def eo_phase_shifter_no_taper(
    modulation_length: float = 7500.0,
    cross_section: CrossSectionSpec = "xs_rwg2000",
    # RF parameters for CPW line
    rf_central_conductor_width: float = 10.0,
    rf_ground_planes_width: float = 180.0,
    rf_gap: float = 4.0,
    cpw_cell: ComponentSpec = uni_cpw_straight,
    draw_cpw: bool = True,
) -> gf.Component:
    """
    Phase shifter with a constant rib waveguide width (no tapers), intended
    for the custom_mzm. The waveguide is located within the gap of a CPW
    transmission line.
    """
    ps = gf.Component()
    xs_modulator = gf.get_cross_section(cross_section)

    # The phase modulation section is a simple straight waveguide
    wg_phase_modulation = gf.components.straight(
        length=modulation_length, cross_section=xs_modulator
    )
    wg_ref = ps << wg_phase_modulation

    ps.add_port(name="o1", port=wg_ref.ports["o1"])
    ps.add_port(name="o2", port=wg_ref.ports["o2"])

    # Add the transmission line (CPW)
    if draw_cpw:
        xs_cpw = gf.partial(
            xs_uni_cpw,
            central_conductor_width=rf_central_conductor_width,
            ground_planes_width=rf_ground_planes_width,
            gap=rf_gap,
        )
        tl = ps << cpw_cell(
            length=modulation_length,
            cross_section=xs_cpw,
            gap_width=rf_gap,
            signal_width=rf_central_conductor_width,
            ground_planes_width=rf_ground_planes_width,
        )

        gap_eff = rf_gap + 2 * np.sum(
            [tl.cell.settings[key] for key in ("tt", "th") if key in tl.cell.settings]
        )

        tl.dmove(
            tl.ports["e1"].dcenter,
            (0.0, -0.5 * rf_central_conductor_width - 0.5 * gap_eff),
        )

        for name, port in [("e1", tl.ports["bp1"]), ("e2", tl.ports["bp2"])]:
            ps.add_port(name=name, port=port)

    ps.flatten()
    return ps


## custom mzm function
@gf.cell
def _custom_mzm_interferometer(
        modulation_length: float = 7500.0,
        length_imbalance: float = 100.0,
        bias_tuning_section_length: float = 750.0,
        sbend_large_size: tuple[float, float] = (200.0, 50.0),
        sbend_small_size: tuple[float, float] = (200.0, -45.0),
        sbend_small_straight_extend: float = 5.0,
        lbend_tune_arm_reff: float = 75.0,
        lbend_combiner_reff: float = 80.0,
) -> gf.Component:
    """
    Custom MZM interferometer core with constant 2um rib width, without MMI
    splitter/combiner. Updated to calculate arm spacing based on custom_mmi_AQ.
    """
    interferometer = gf.Component()
    xs_custom = "xs_rwg2000"

    # Get MMI component info for positioning, but do not place the component.
    # This now uses custom_mmi_AQ to get the correct port separation.
    splitter_combiner_cell = custom_mmi_AQ()
    port_separation_y = abs(
        splitter_combiner_cell.ports["o3"].dy - splitter_combiner_cell.ports["o2"].dy
    )

    # --- Sub-component Definitions ---
    sbend_large = S_bend_vert(
        v_offset=sbend_large_size[1],
        h_extent=sbend_large_size[0],
        dx_straight=5.0,
        cross_section=xs_custom,
    )
    sbend_small = S_bend_vert(
        v_offset=sbend_small_size[1],
        h_extent=sbend_small_size[0],
        dx_straight=sbend_small_straight_extend,
        cross_section=xs_custom,
    )

    def branch_top():
        bt = gf.Component()
        sbend_1 = bt << sbend_large
        sbend_2 = bt << sbend_small
        pm = bt << eo_phase_shifter_no_taper(
            modulation_length=modulation_length, draw_cpw=False, cross_section=xs_custom
        )
        sbend_3 = bt << sbend_small
        sbend_2.connect("o1", sbend_1.ports["o2"])
        pm.connect("o1", sbend_2.ports["o2"])
        sbend_3.dmirror_x()
        sbend_3.connect("o1", pm.ports["o2"])
        bt.add_port(name="o1", port=sbend_1.ports["o1"])
        bt.add_port(name="o2", port=sbend_3.ports["o2"])
        bt.add_port(name="phase_shifter_start", port=pm.ports["o1"])
        bt.flatten()
        return bt

    def branch_tune_short(straight_unbalance: float = 0.0):
        lbend = L_turn_bend(radius=lbend_tune_arm_reff, cross_section=xs_custom)
        straight_y = gf.components.straight(
            length=20.0 + straight_unbalance, cross_section=xs_custom
        )
        straight_x = gf.components.straight(
            length=bias_tuning_section_length, cross_section=xs_custom
        )
        symbol_to_component = {
            "b": (lbend, "o1", "o2"),
            "L": (straight_y, "o1", "o2"),
            "B": (lbend, "o2", "o1"),
            "_": (straight_x, "o1", "o2"),
        }
        sequence = "bLB_!b!L"
        arm = gf.components.component_sequence(
            sequence=sequence,
            ports_map={"phase_tuning_segment_start": ("_1", "o1")},
            symbol_to_component=symbol_to_component,
        )
        arm.add_port(port=arm.ports["phase_tuning_segment_start"])
        arm.flatten()
        return arm

    def branch_tune_long(straight_unbalance):
        return partial(branch_tune_short, straight_unbalance=straight_unbalance)()

    # --- Component Placement (MMIs are NOT placed) ---
    bt = interferometer << branch_top()
    bb = interferometer << branch_top()
    bs = interferometer << branch_tune_short()
    bl = interferometer << branch_tune_long(abs(0.5 * length_imbalance))
    lbend_c = L_turn_bend(radius=lbend_combiner_reff, cross_section=xs_custom)
    lbend_top = interferometer << lbend_c
    lbend_bottom = interferometer << lbend_c

    # --- Routing and Connections ---
    # 1. Manually place the arms where the splitter would have put them
    bb.dmirror_y()
    bt.dmove(origin=bt.ports["o1"].dcenter, destination=(0, port_separation_y / 2))
    bb.dmove(origin=bb.ports["o1"].dcenter, destination=(0, -port_separation_y / 2))

    # 2. Modulator arms to bias tuning arms
    if length_imbalance >= 0:
        bl.connect("o1", bt.ports["o2"])
        bs.dmirror_y()
        bs.connect("o1", bb.ports["o2"])
    else:
        bs.connect("o1", bt.ports["o2"])
        bl.dmirror_y()
        bl.connect("o1", bb.ports["o2"])

    # 3. Bias tuning arms to final L-bends
    lbend_bottom.dmirror_y()
    if length_imbalance >= 0:
        lbend_top.connect("o1", bl.ports["o2"])
        lbend_bottom.connect("o1", bs.ports["o2"])
    else:
        lbend_top.connect("o1", bs.ports["o2"])
        lbend_bottom.connect("o1", bl.ports["o2"])

    # --- Port Exposure ---
    exposed_ports = [
        ("in_top", bt.ports["o1"]),
        ("in_bot", bb.ports["o1"]),
        ("out_top", lbend_top.ports["o2"]),
        ("out_bot", lbend_bottom.ports["o2"]),
        ("upper_phase_shifter_start", bt.ports["phase_shifter_start"]),
        ("short_bias_branch_start", bs.ports["phase_tuning_segment_start"]),
        ("long_bias_branch_start", bl.ports["phase_tuning_segment_start"]),
    ]
    for name, port in exposed_ports:
        interferometer.add_port(name=name, port=port)
    interferometer.flatten()
    return interferometer


@gf.cell
def custom_mzm(
        modulation_length: float = 7500.0,
        length_imbalance: float = 100.0,
        lbend_tune_arm_reff: float = 75.0,
        rf_pad_start_width: float = 80.0,
        rf_central_conductor_width: float = 10.0,
        rf_ground_planes_width: float = 180.0,
        rf_gap: float = 4.0,
        rf_pad_length_straight: float = 10.0,
        rf_pad_length_tapered: float = 300.0,
        bias_tuning_section_length: float = 700.0,
        cpw_cell: ComponentSpec = uni_cpw_straight,
        with_heater: bool = False,
        heater_offset: float = 1.2,
        heater_width: float = 1.0,
        heater_pad_size: tuple[float, float] = (75.0, 75.0),
) -> gf.Component:
    """
    Custom Mach-Zehnder modulator with constant 2um rib width and no tapers.
    The MMI splitter/combiner are removed and the optical ports are exposed.
    The geometric calculations have been updated for custom_mmi_AQ.
    """
    mzm = gf.Component()

    # --- RF Transmission Line Subcell ---
    xs_cpw = gf.partial(
        xs_uni_cpw,
        central_conductor_width=rf_central_conductor_width,
        ground_planes_width=rf_ground_planes_width,
        gap=rf_gap,
    )
    rf_line = mzm << cpw_cell(
        bondpad={
            "component": "CPW_pad_linear",
            "settings": {
                "start_width": rf_pad_start_width,
                "length_straight": rf_pad_length_straight,
                "length_tapered": rf_pad_length_tapered,
            },
        },
        length=modulation_length,
        signal_width=rf_central_conductor_width,
        cross_section=xs_cpw,
        ground_planes_width=rf_ground_planes_width,
        gap_width=rf_gap,
    )
    rf_line.dmove(rf_line.ports["e1"].dcenter, (0.0, 0.0))

    # --- Interferometer Subcell ---
    # Instantiate the target MMI to get its port separation for calculations.
    splitter = custom_mmi_AQ()
    port_separation = abs(splitter.ports["o2"].dy - splitter.ports["o3"].dy)

    sbend_large_AR = 6
    gap_eff = rf_gap + 2 * np.sum(
        [
            rf_line.cell.settings[key]
            for key in ("tt", "th")
            if key in rf_line.cell.settings
        ]
    )
    GS_separation = rf_pad_start_width * gap_eff / rf_central_conductor_width

    # This calculation is now based on the actual port separation of custom_mmi_AQ.
    sbend_large_v_offset = (
            0.5 * rf_pad_start_width
            + 0.5 * GS_separation
            - 0.5 * port_separation
    )

    sbend_small_straight_length = rf_pad_length_straight * 0.5

    # This calculation is also updated.
    lbend_combiner_reff = (
            0.5 * rf_pad_start_width
            + lbend_tune_arm_reff
            + 0.5 * GS_separation
            - 0.5 * port_separation
    )

    interferometer = mzm << _custom_mzm_interferometer(
        modulation_length=modulation_length,
        length_imbalance=length_imbalance,
        sbend_large_size=(
            sbend_large_AR * sbend_large_v_offset,
            sbend_large_v_offset,
        ),
        sbend_small_size=(
            rf_pad_length_straight
            + rf_pad_length_tapered
            - 2 * sbend_small_straight_length,
            -0.5
            * (
                    rf_pad_start_width
                    - rf_central_conductor_width
                    + GS_separation
                    - gap_eff
            ),
        ),
        sbend_small_straight_extend=sbend_small_straight_length,
        lbend_tune_arm_reff=lbend_tune_arm_reff,
        lbend_combiner_reff=lbend_combiner_reff,
        bias_tuning_section_length=bias_tuning_section_length,
    )

    interferometer.dmove(
        interferometer.ports["upper_phase_shifter_start"].dcenter,
        (0.0, 0.5 * (rf_central_conductor_width + gap_eff)),
    )

    # --- Heater for Phase Tuning (Optional) ---
    if with_heater:
        ht_ref = mzm << heater_straight_single(
            length=bias_tuning_section_length,
            width=heater_width,
            offset=heater_offset,
            pad_size=heater_pad_size,
        )
        if length_imbalance < 0.0:
            heater_disp = [0, 0.5 * heater_width + heater_offset]
        else:
            ht_ref.dmirror_y()
            heater_disp = [0, -0.5 * heater_width - heater_offset]
        ht_ref.dmove(
            origin=ht_ref.ports["ht_start"].dcenter,
            destination=(
                    np.array(interferometer.ports["long_bias_branch_start"].dcenter)
                    + heater_disp
            ),
        )

    # --- Port Exposure ---
    exposed_ports = [
        ("e1", rf_line.ports["bp1"]),
        ("e2", rf_line.ports["bp2"]),
        ("in_top", interferometer.ports["in_top"]),
        ("in_bot", interferometer.ports["in_bot"]),
        ("out_top", interferometer.ports["out_top"]),
        ("out_bot", interferometer.ports["out_bot"]),
    ]
    if with_heater:
        exposed_ports += [
            ("e3", ht_ref.ports["e1"]),
            ("e4", ht_ref.ports["e2"]),
        ]
    [mzm.add_port(name=name, port=port) for name, port in exposed_ports]

    return mzm


##custom mmi
@gf.cell
def custom_mmi(
    width_mmi: float = 15,
    length_mmi: float = 60,
    width_taper: float = 1.5,
    length_taper: float = 25.0,
    port_ratio: float = 0.55,
    cross_section: CrossSectionSpec = "xs_rwg2000",
    **kwargs,
) -> gf.Component:
    """MMI1x2 with layout optimized for maximum transmission at 1550 nm."""

    gap_mmi = (
        port_ratio * width_mmi - width_taper
    )  # The port ratio is defined as the ratio between the waveguides separation and the MMI width.

    return gf.components.mmi1x2(
        width_mmi=width_mmi,
        length_mmi=length_mmi,
        gap_mmi=gap_mmi,
        length_taper=length_taper,
        width_taper=width_taper,
        cross_section=cross_section,
        **kwargs,
    )

##  custom phase eo phase shifter
@gf.cell
def single_custom_ps(
    modulation_length: float = 3000.0,
    rib_width: float = 2.0,
    rf_central_conductor_width: float = 10.0,
    rf_gap: float = 4.5,
    rf_ground_planes_width: float = 180.0,
) -> gf.Component:
    """
    Creates a dual-waveguide EO shifter with a constant waveguide width.

    This component features two parallel optical waveguides with no tapers,
    placed symmetrically within the gaps of a single CPW transmission line.
    """
    c = gf.Component()

    # 1. Instantiate the single RF Transmission Line (CPW)
    rf_line = c << lnoi400.cells.uni_cpw_straight(
        length=modulation_length,
        signal_width=rf_central_conductor_width,
        gap_width=rf_gap,
        ground_planes_width=rf_ground_planes_width,
    )

    # 2. Define the cross-section for the optical waveguides
    # We get the standard rib waveguide cross-section and set a new width.
    xs_waveguide = gf.get_cross_section("xs_rwg2000")

    # 3. Create the optical arms as simple straight waveguides (no tapers)
    optical_arm = gf.components.straight(
        length=modulation_length, cross_section=xs_waveguide
    )

    arm_top = c << optical_arm
    # arm_bottom = c << optical_arm

    # 4. Position the optical arms in the CPW gaps
    y_offset = 0.5 * (rf_central_conductor_width + rf_gap)
    rf_line.dcenter = (0, 0)
    arm_top.dcenter = (rf_line.dcenter[0], y_offset)
    # arm_bottom.dcenter = (rf_line.dcenter[0], -y_offset)

    # 5. Add ports to the top-level component
    c.add_ports(rf_line.ports)
    c.add_ports(arm_top.ports, prefix="top_")
    # c.add_ports(arm_bottom.ports, prefix="bot_")

    # c.flatten()
    return c

## Concentric rings for soliton
@gf.cell
def concentric_rings_with_bus(
    radius_inner: float = 230.0,
    gap_inner_outer: float = 6.3,
    coupling_gap_bus: float = 3,
    bus_length: float = 400.0,
) -> gf.Component:
    """
    Creates a component with two concentric rings and a coupled bus waveguide.

    Args:
        radius_inner: Radius of the inner ring path.
        gap_inner_outer: Edge-to-edge gap between inner and outer rings.
        coupling_gap_bus: Edge-to-edge gap between the outer ring and the bus.
        bus_length: Length of the straight bus waveguide.
    """
    c = gf.Component("resonator_with_bus")

    # --- Cross-section Definitions ---
    xs_inner = gf.get_cross_section('xs_rwg5000')
    xs_outer = gf.get_cross_section('xs_rwg1380', width=1.380)
    xs_bus = gf.get_cross_section('xs_rwg2000')

    # --- Inner Ring ---
    path_inner = arc(radius=radius_inner, angle=360)
    inner_ring = c << path_inner.extrude(cross_section=xs_inner)

    # --- Outer Ring ---
    # Calculate radius using widths from cross-sections for accuracy
    radius_outer_center = (
        radius_inner
        + (xs_inner.width / 2)
        + gap_inner_outer
        + (xs_outer.width / 2)
    )
    path_outer = arc(radius=radius_outer_center, angle=360)
    outer_ring = c << path_outer.extrude(cross_section=xs_outer)
    outer_ring.center = inner_ring.center

    # --- Bus Waveguide ---
    bus_wg = straight_rwg2000(length=bus_length)
    bus_ref = c << bus_wg

    # --- Position the bus waveguide at the BOTTOM ---
    # Correctly calculate the vertical position for the bus waveguide's center
    y_bus_center = -(
        5/2+gap_inner_outer+1.38+1+coupling_gap_bus
    )

    # Move the bus waveguide to the calculated position.
    bus_ref.movex(-bus_length/2)
    bus_ref.dmovey(y_bus_center)

    # Add ports from the bus waveguide to the final component
    c.add_ports(bus_ref.ports)

    return c

##################################################################################### End: Redwan Islam