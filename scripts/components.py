import gdsfactory as gf
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec
from lnoi400.tech import LAYER, xs_uni_cpw
from lnoi400.spline import bend_S_spline_varying_width
import numpy as np

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path

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