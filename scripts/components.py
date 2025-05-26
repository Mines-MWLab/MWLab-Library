import gdsfactory as gf
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec
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

    wg2.connect("o2",wg1.ports["o2"])
    wg3.connect("o1", wg1.ports["o1"])

    exposed_ports = [
        ("o1", wg2.ports["o1"]),
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
            0.5*wg3.ports["o1"].center[1]+ 0.5*wg2.ports["o2"].center[1])



    rt.dmovex(rt.ports["o1"].center[0],
            wg3.ports["o2"].center[0]+sizex_bus-0.5*size_buswg-size_buswg-sizex_rt-pulley_gap,
            )

    [c.add_port(name=name, port=port) for name, port in [ ("o3", rt.ports["o1"]),("o4", rt.ports["o2"])]]
    return c

