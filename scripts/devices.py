import gdsfactory as gf
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec, ComponentSpec
from lnoi400.spline import bend_S_spline_varying_width
import numpy as np
import components as mpl

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path

#####################
# Traveling Wave EOM (final design)
#####################
@gf.cell
def tWave_EOM(
    # trail_cpw variables
    rf_central_conductor_width: float = 21.0,
    rf_gap: float = 4,
    modulation_length: float = 4330.0,

    # mzm_unbalanced variables
    length_imbalance: float = 0.0,
    lbend_tune_arm_reff: float = 75.0,
    rf_pad_start_width: float = 80.0,
    #rf_ground_planes_width: float = 180.0,
    rf_pad_length_straight: float = 10.0,
    rf_pad_length_tapered: float = 300.0,
    bias_tuning_section_length: float = 700.0,
    cpw_cell: ComponentSpec = lnoi400.cells.trail_cpw,
    with_heater: bool = True,
    heater_offset: float = 1.2,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),

    # #_mzm_interferometer variables
    taper_length: float = 100.0,
    rib_core_width_modulator: float = 2.5,


) -> gf.Component:

    mzm = gf.Component()
    

    rf_line = mzm << cpw_cell(length = modulation_length,
                                signal_width = rf_central_conductor_width,
                                gap_width = rf_gap,
                                th = 3.0,            #h
                                tl = 44.7,           #r
                                tw = 7.0,            #t
                                tt = 1.5,            #s
                                tc = 5.0,            #c
                                ground_planes_width = 180.0,
                                rounding_radius = 0.5,
                                bondpad={
                                        "component": "CPW_pad_linear",
                                         "settings": {
                                                    "start_width": rf_pad_start_width,
                                                    "length_straight": rf_pad_length_straight,
                                                    "length_tapered": rf_pad_length_tapered,
                                    },
                                })
    # default code
    rf_line.dmove(rf_line.ports["e1"].dcenter, (0.0, 0.0))


    # Interferometer subcell

    splitter1 = lnoi400.cells.mmi1x2_optimized1550()
    splitter2 = lnoi400.cells.mmi2x2_optimized1550()

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
        - 0.5 * splitter1.settings["port_ratio"] * splitter1.settings["width_mmi"]
    )

    sbend_small_straight_length = rf_pad_length_straight * 0.5

    lbend_combiner_reff = (
        0.5 * rf_pad_start_width
        + lbend_tune_arm_reff
        + 0.5 * GS_separation
        - 0.5 * splitter2.settings["port_ratio"] * splitter2.settings["width_mmi"]
    )

    sbend_large_size = (
        sbend_large_AR * sbend_large_v_offset,
        sbend_large_v_offset,
    )
    sbend_small_size = (
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
    )
    sbend_small_straight_extend = sbend_small_straight_length
    lbend_tune_arm_reff = lbend_tune_arm_reff
    lbend_combiner_reff = lbend_combiner_reff
    bias_tuning_section_length = bias_tuning_section_length


    # placing the waveguide components from "_mzm_interferometer" cell manually

    sbend_large = lnoi400.cells.S_bend_vert(
        v_offset=sbend_large_size[1], h_extent=sbend_large_size[0], dx_straight=5.0
    )

    sbend_small = lnoi400.cells.S_bend_vert(
        v_offset=sbend_small_size[1],
        h_extent=sbend_small_size[0],
        dx_straight=sbend_small_straight_extend,
    )

    interferometer = gf.Component()

    def branch_top():
        bt = gf.Component()
        sbend_1 = bt << sbend_large
        sbend_2 = bt << sbend_small
        pm = bt << lnoi400.cells.eo_phase_shifter_high_speed(
            rib_core_width_modulator=rib_core_width_modulator,
            modulation_length=modulation_length,
            taper_length=taper_length,
            draw_cpw=False,
        )
        sbend_3 = bt << sbend_small
        sbend_2.connect("o1", sbend_1.ports["o2"])
        pm.connect("o1", sbend_2.ports["o2"])
        sbend_3.dmirror_x()
        sbend_3.connect("o1", pm.ports["o2"])

        for name, port in [
            ("o1", sbend_1.ports["o1"]),
            ("o2", sbend_3.ports["o2"]),
            ("taper_start", pm.ports["o1"]),
        ]:
            bt.add_port(name=name, port=port)
        bt.flatten()

        return bt

    def branch_tune_short(straight_unbalance: float = 0.0):
        arm = gf.Component()
        lbend = lnoi400.cells.L_turn_bend(radius=lbend_tune_arm_reff)
        straight_y = gf.components.straight(
            length=20.0 + straight_unbalance, cross_section="xs_rwg1000"
        )
        straight_x = gf.components.straight(
            length=bias_tuning_section_length, cross_section="xs_rwg1000"
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
    

    #forcing the option for a 1x2 splitter on the input and a 2x2 splitter on the output
    #splt = gf.get_component(splitter2)
    splt1 = gf.get_component(splitter1)
    splt2 = gf.get_component(splitter2)

    # Uniformly handle the cases of a 1x2 or 2x2 MMI

    out_top = splt1.ports["o2"]
    out_bottom = splt1.ports["o3"]

    # at output
    out_top_out = splt2.ports["o3"]
    out_bottom_out = splt2.ports["o4"]


    def combiner_section():
        comb_section = gf.Component()
        lbend_combiner = lnoi400.cells.L_turn_bend(radius=lbend_combiner_reff)
        lbend_top = comb_section << lbend_combiner
        lbend_bottom = comb_section << lbend_combiner
        lbend_bottom.dmirror_y()

        #pushes splitter2 to design
        combiner = comb_section << splt2
        #combiner = comb_section << splt
        lbend_top.connect("o1", out_top_out)
        lbend_bottom.connect("o1", out_bottom_out)

        # comb_section.flatten()

        exposed_ports = [
            ("o2", lbend_top.ports["o2"]),
            ("o1", combiner.ports["o1"]),
            ("o3", lbend_bottom.ports["o2"]),
            ("o4", combiner.ports["o2"]),
        ]


        for name, port in exposed_ports:
            comb_section.add_port(name=name, port=port)

        #comb_section.show()

        return comb_section

    # pushes splitter1 to design
    splt_ref = interferometer << splt1
    
    bt = interferometer << branch_top()
    bb = interferometer << branch_top()
    bs = interferometer << branch_tune_short()
    bl = interferometer << branch_tune_long(abs(0.5 * length_imbalance))
    cs = interferometer << combiner_section()
    bb.dmirror_y()
    bt.connect("o1", out_top)
    bb.connect("o1", out_bottom)
    if length_imbalance >= 0:
        bs.dmirror_y()
        bs.connect("o1", bb.ports["o2"])
        bl.connect("o1", bt.ports["o2"])
    else:
        bs.connect("o1", bt.ports["o2"])
        bl.dmirror_y()
        bl.connect("o1", bb.ports["o2"])
    cs.dmirror_x()
    [
        cs.connect("o2", bl.ports["o2"])
        if length_imbalance >= 0
        else cs.connect("o2", bs.ports["o2"])
    ]


    #splt1.show()
    vaportvar = bt.ports["taper_start"].dcenter
    print(vaportvar)

    interferometer.dmove(
        vaportvar,
        (0.0, 0.5 * (rf_central_conductor_width + gap_eff)),
    )

    exposed_ports = [
        ("o1", splt_ref.ports["o1"]),
        ("upper_taper_start", bt.ports["taper_start"]),
        ("short_bias_branch_start", bs.ports["phase_tuning_segment_start"]),
        ("long_bias_branch_start", bl.ports["phase_tuning_segment_start"]),
        ("o2", cs.ports["o4"]),
        ("o3", cs.ports["o1"]),
    ]

    for name, port in exposed_ports:
        interferometer.add_port(name=name, port=port)



    interferometer.flatten()

    interferometer_ref = mzm << interferometer


    if with_heater:
        ht_ref = mzm << lnoi400.cells.heater_straight_single(
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
                np.array(interferometer_ref.ports["long_bias_branch_start"].dcenter)
                + heater_disp
            ),
        )

    # Expose the ports

    exposed_ports += [
        ("e1", rf_line.ports["bp1"]),
        ("e2", rf_line.ports["bp2"]),
    ]

    if with_heater:
        exposed_ports += [
            ("e3", ht_ref.ports["e1"]),
            (
                "e4",
                ht_ref.ports["e2"],
            ),
        ]

    [mzm.add_port(name=name, port=port) for name, port in exposed_ports]

    
    return mzm

#####################
# Dual Optically Resonant EOM w/ GSG landing (draft design)
#####################
@gf.cell
def dOR_EOM():

    #define constants for race track
    h_racetrack = 200.0
    ls = 335.0
    modulation_length = ls
    lextra = 50

    # trail_cpw variables
    
    rf_gap = 4
    h = 3.0
    r = 44.7
    t = 7.0
    s = 1.5
    c = 5.0

    rf_central_conductor_width = 21.0

    # taken from source code, find number of tcells to calculate ls
    #num_cells = np.floor(modulation_length / (tl + tc))

    #rf_central_conductor_width = h_racetrack - rf_gap - 2*s - 2*h     # w, adjusted for optical waveguide bend radius
    #h_racetrack = rf_gap + 2*s + 2*h + rf_central_conductor_width

    #define subcomponents
    u_bend_racetrack = lnoi400.cells.U_bend_racetrack(
        v_offset = h_racetrack,
        p = 1.0,
        with_arc_floorplan = True,
        cross_section = "xs_rwg3000",
    )

    dc_wg = gf.components.straight(
        length = ls+2*lextra,
        cross_section="xs_rwg3000",
    )

    # create the Tcell component with NO GSG LANDING
    rf_line = lnoi400.cells.trail_cpw(
        length = modulation_length,
        signal_width = rf_central_conductor_width,
        gap_width = rf_gap,
        th = h,            #h
        tl = r,            #r
        tw = t,            #t
        tt = s,            #s
        tc = c,            #c
        ground_planes_width = 180.0,
        rounding_radius = 0.5,
        bondpad = {
                    "component": "CPW_pad_linear",
                    "settings": {
                    "start_width": 80.0,
                    "length_straight": 10.0,
                    "length_tapered": 190.0,
            },
        }
        )
    
    def racetrack():
        # push subcomponents to design
        race_track = gf.Component()
        u_bend_ref1 = race_track << u_bend_racetrack
        u_bend_ref2 = race_track << u_bend_racetrack
        straight_ref1 = race_track << dc_wg
        straight_ref2 = race_track << dc_wg


        u_bend_ref1.connect("o2", straight_ref1.ports["o2"])
        straight_ref2.connect("o2", u_bend_ref1.ports["o1"])
        u_bend_ref2.connect("o1", straight_ref1.ports["o1"])
        u_bend_ref2.connect("o2", straight_ref2.ports["o1"])

        #race_track.flatten()

        return race_track

    
    # create assembly
    c = gf.Component()

    race_track_top = c << racetrack()
    race_track_bottom = c << racetrack()
    rf_line_ref = c << rf_line

    #define position of rf_line

    gap_eff = rf_gap + 2 * np.sum(
        [rf_line_ref.cell.settings[key] for key in ("tt", "th") if key in rf_line_ref.cell.settings]
    )
    #print(gap_eff)

    rf_line_ref.dmove(
        rf_line_ref.ports["e1"].dcenter,
        (lextra,0.0),
    )

    race_track_top.dmove((0, h_racetrack + rf_central_conductor_width/2 + s + h + rf_gap/2))
    race_track_bottom.dmove((0, -(rf_central_conductor_width/2 + s + h + rf_gap/2)))

    #Add rf_line ports to system
    for name, port in [
        ("e1", rf_line_ref.ports["bp1"]),
        ("e2", rf_line_ref.ports["bp2"]),
    ]:
        c.add_port(name=name, port=port)   

    #print(rf_line.get_ports_list())
    #print(h_racetrack)

    return c



#####################
# Dual Optically Resonant EOM w/edge couplers and GSG landing (draft design)
#####################
@gf.cell
def dOR_EOM_DC(
    RT_cross_section: CrossSectionSpec = "xs_rwg3000",
    DC_cross_section: CrossSectionSpec = "xs_rwg1000",
    DC_io_wg_sep: float = 25,#15.3,
    DC_sbend_length: float = 150,
    DC_central_straight_length: float = 16.92,
    DC_coupl_wg_sep: float = 0.6,
    DC_coup_wg_width: float = 1.2,
    DC_ubend_sep: float = 100,
    h_racetrack: float = 200.0,
    ls: float = 1408.0, #3850
    lextra: float = 364, #205
    rf_gap:float = 4,
    rf_central_conductor_width: float = 21.0,
    h: float = 3.0,
    r: float = 44.7,
    t: float = 7.0,
    s: float = 1.5,
    c: float = 5.0,
    start_width: float = 80.0,
    length_straight: float = 10.0,
    length_tapered: float = 190.0,
)-> gf.Component:

    #define constants for race track
    #h_racetrack = 200.0
    #ls = 335.0
    modulation_length = ls
    


    # trail_cpw variables
    
    #rf_gap = 4
    #h = 3.0
    #r = 44.7
    #t = 7.0
    #s = 1.5
    #c = 5.0

    #rf_central_conductor_width = 21.0

    # taken from source code, find number of tcells to calculate ls
    #num_cells = np.floor(modulation_length / (tl + tc))

    #rf_central_conductor_width = h_racetrack - rf_gap - 2*s - 2*h     # w, adjusted for optical waveguide bend radius
    #h_racetrack = rf_gap + 2*s + 2*h + rf_central_conductor_width

    #define subcomponents
    u_bend_racetrack = lnoi400.cells.U_bend_racetrack(
        v_offset = h_racetrack,
        p = 1.0,
        with_arc_floorplan = True,
        cross_section = RT_cross_section,
    )

    ecoup = mpl.asymmetric_directional_coupler_racetrack(
        cross_section_io=RT_cross_section,
        cross_section_bus=DC_cross_section,
        io_wg_sep = DC_io_wg_sep*2,
        sbend_length = DC_sbend_length,
        central_straight_length = DC_central_straight_length,
        coupl_wg_sep = DC_coupl_wg_sep,
        coup_wg_width = DC_coup_wg_width,
    )
    ec_xsize = ecoup.xsize

    dc_wg_sep = gf.components.straight(
        length = DC_ubend_sep,
        cross_section=RT_cross_section,
    )
    dc_wg_shorter = gf.components.straight(
        length = ls+2*lextra-ec_xsize-DC_ubend_sep,
        cross_section=RT_cross_section,
    )

    dc_wg = gf.components.straight(
        length = ls+2*lextra,
        cross_section=RT_cross_section,
    )



    # create the Tcell component with NO GSG LANDING
    rf_line = mpl.trail_cpw_mpl(
        length = modulation_length,
        signal_width = rf_central_conductor_width,
        gap_width = rf_gap,
        th = h,            #h
        tl = r,            #r
        tw = t,            #t
        tt = s,            #s
        tc = c,            #c
        ground_planes_width = 180.0,
        rounding_radius = 0.5,
        bondpad = {
                    "component": "CPW_pad_linear",
                    "settings": {
                    "start_width": start_width,
                    "length_straight": length_straight,
                    "length_tapered": length_tapered,
            },
        }
        )

    def racetrack():
        # push subcomponents to design
        race_track = gf.Component()
        u_bend_ref1 = race_track << u_bend_racetrack
        u_bend_ref2 = race_track << u_bend_racetrack
        straight_ref1 = race_track << dc_wg_shorter
        straight_sep_ref = race_track << dc_wg_sep
        ec_top = race_track << ecoup
        straight_ref2 = race_track << dc_wg


        u_bend_ref1.connect("o2", straight_ref1.ports["o2"])
        straight_ref2.connect("o2", u_bend_ref1.ports["o1"])
        ec_top.connect("o3", straight_ref1["o1"])
        straight_sep_ref.connect("o1", ec_top.ports["o4"])
        u_bend_ref2.connect("o1", straight_sep_ref.ports["o2"])
        #u_bend_ref2.connect("o1", ec_top.ports["o4"])
        u_bend_ref2.connect("o2", straight_ref2.ports["o1"])

        exposed_ports = [
            ("o1", ec_top.ports["o1"]),
            ("o2", ec_top.ports["o2"])
        ]
        [race_track.add_port(name=name, port=port) for name, port in exposed_ports]
        #race_track.flatten()

        return race_track

    
    # create assembly
    c = gf.Component()

    race_track_top = c << racetrack()
    race_track_bottom = c << racetrack()
    race_track_bottom.dmirror_y()
    rf_line_ref = c << rf_line
    
    #define position of rf_line

    gap_eff = rf_gap + 2 * np.sum(
        [rf_line_ref.cell.settings[key] for key in ("tt", "th") if key in rf_line_ref.cell.settings]
    )
    #print(gap_eff)

    rf_line_ref.dmove(
        (rf_line_ref.ports["e1"].dcenter[0]*0.5 + rf_line_ref.ports["e2"].dcenter[0]*0.5,0),
        (race_track_top.center[0],0.0),
    )

    race_track_top.dmove((0, h_racetrack + rf_central_conductor_width/2 + s + h + rf_gap/2))
    race_track_bottom.dmove((0, -h_racetrack-(rf_central_conductor_width/2 + s + h + rf_gap/2)))

    #Add rf_line ports to system
    for name, port in [
        ("e1", rf_line_ref.ports["bp1"]),
        ("e2", rf_line_ref.ports["bp2"]),
    ]:
        c.add_port(name=name, port=port)   
    
    # Expose the ports
    exposed_ports = [
        ("o1", race_track_top.ports["o1"]),
        ("o2", race_track_top.ports["o2"]),
        ("o3", race_track_bottom.ports["o1"]),
        ("o4", race_track_bottom.ports["o2"]),
    ]

    [c.add_port(name=name, port=port) for name, port in exposed_ports]

    return c


#####################
# EO comb
#####################
@gf.cell
def EOcomb(
    RT_cross_section: CrossSectionSpec = "xs_rwg3000",
    DC_cross_section: CrossSectionSpec = "xs_rwg1000",
    DC_io_wg_sep: float = 25,#15.3,
    DC_sbend_length: float = 150,
    DC_central_straight_length: float = 16.92,
    DC_coupl_wg_sep: float = 0.6,
    DC_coup_wg_width: float = 1.2,
    DC_ubend_sep: float = 100,
    h_racetrack: float = 200.0,
    ls: float = 1788, #3850
    rf_gap:float = 4,
    rf_central_conductor_width: float = 21.0,
    h: float = 3.0,
    r: float = 44.7,
    t: float = 7.0,
    s: float = 1.5,
    c: float = 5.0,
    RF_res_layout_path = None,
    RF_pad_layout_path = None,
    RF_pad_pos = "right",
    RF_pad_gap:float  = 15,
    RF_flipy: bool = False,
    opt_flipy: bool = False,
)-> gf.Component:

    lextra = 0
    #define subcomponents
    u_bend_racetrack = lnoi400.cells.U_bend_racetrack(
        v_offset = h_racetrack,
        p = 1.0,
        with_arc_floorplan = True,
        cross_section = RT_cross_section,
    )

    ecoup = mpl.asymmetric_directional_coupler_racetrack(
        cross_section_io=RT_cross_section,
        cross_section_bus=DC_cross_section,
        io_wg_sep = DC_io_wg_sep*2,
        sbend_length = DC_sbend_length,
        central_straight_length = DC_central_straight_length,
        coupl_wg_sep = DC_coupl_wg_sep,
        coup_wg_width = DC_coup_wg_width,
    )
    ec_xsize = ecoup.xsize

    dc_wg_sep = gf.components.straight(
        length = DC_ubend_sep,
        cross_section=RT_cross_section,
    )
    dc_wg_shorter = gf.components.straight(
        length = ls+2*lextra-ec_xsize-DC_ubend_sep,
        cross_section=RT_cross_section,
    )

    dc_wg = gf.components.straight(
        length = ls+2*lextra,
        cross_section=RT_cross_section,
    )


    def racetrack():
        # push subcomponents to design
        race_track = gf.Component()
        u_bend_ref1 = race_track << u_bend_racetrack
        u_bend_ref2 = race_track << u_bend_racetrack
        straight_ref1 = race_track << dc_wg_shorter
        straight_sep_ref = race_track << dc_wg_sep
        ec_top = race_track << ecoup
        straight_ref2 = race_track << dc_wg


        u_bend_ref1.connect("o2", straight_ref1.ports["o2"])
        straight_ref2.connect("o2", u_bend_ref1.ports["o1"])
        ec_top.connect("o3", straight_ref1["o1"])
        straight_sep_ref.connect("o1", ec_top.ports["o4"])
        u_bend_ref2.connect("o1", straight_sep_ref.ports["o2"])
        #u_bend_ref2.connect("o1", ec_top.ports["o4"])
        u_bend_ref2.connect("o2", straight_ref2.ports["o1"])

        exposed_ports = [
            ("o1", ec_top.ports["o1"]),
            ("o2", ec_top.ports["o2"]),
            ("ubend1", u_bend_ref1.ports["o1"]),
            ("ubend2", u_bend_ref1.ports["o2"])
        ]
        [race_track.add_port(name=name, port=port) for name, port in exposed_ports]

        return race_track

    # create assembly
    c = gf.Component()
    rt_c = racetrack()
    race_track_top = c << rt_c

    rfpadgap = RF_pad_gap
    if RF_res_layout_path:
        mw = gf.read.import_gds(RF_res_layout_path)
        layer_map = {(1, 0): (21, 0)}
        mw.remap_layers(layer_map)    
        c_dx = c.dx
        c_dy = 0.5*race_track_top.ports["ubend1"].dy + 0.5*race_track_top.ports["ubend2"].dy
        crf = c << mw
        crf.dmovex(crf.dx, c_dx)
        crf.dmovey(crf.dy, c_dy)
        crf.dmirror_y()
        if RF_pad_layout_path:
            pad = gf.read.import_gds(RF_pad_layout_path)
            pad.remap_layers(layer_map)
            rf_xsize = crf.xsize
            rf_ysize = crf.ysize
            cpad = c << pad
            if RF_pad_pos == "right":
                cpad.dmovex(cpad.dx, c_dx + rf_xsize/2 + cpad.xsize/2+rfpadgap)
                cpad.dmovey(cpad.dy, c_dy)
                cpad.dmirror_y()
            elif RF_pad_pos == "bottom":
                cpad.dmovex(cpad.dx, c_dx)
                cpad.dmovey(cpad.dy, c_dy - rf_ysize/2- cpad.ysize/2 -rfpadgap)
                cpad.dmirror_y()
        
    race_track_top.dmirror_y()
    if RF_flipy:
        c_dy_upd = 0.5*race_track_top.ports["ubend1"].dy + 0.5*race_track_top.ports["ubend2"].dy
        if RF_res_layout_path:
            cpad.dmirror_y(c_dy_upd)
            if RF_pad_layout_path:
                crf.dmirror_y(c_dy_upd)
        
    if opt_flipy:
        race_track_top.dmirror_y(0.5*race_track_top.ports["ubend1"].dy + 0.5*race_track_top.ports["ubend2"].dy)
    

    # Expose the ports
    exposed_ports = [
        ("o1", race_track_top.ports["o1"]),
        ("o2", race_track_top.ports["o2"]),
        #("ubend1", race_track_top.ports["ubend1"]),
        #("ubend2", race_track_top.ports["ubend2"]),
    ]

    [c.add_port(name=name, port=port) for name, port in exposed_ports]
    c.flatten()
    return c



#####################
# Dual EO comb
#####################
@gf.cell
def dualEOcomb(
    comb_sep: float = 101.25+50,
    RT_cross_section: CrossSectionSpec = "xs_rwg3000",
    DC_cross_section: CrossSectionSpec = "xs_rwg1000",
    DC_io_wg_sep: float = 25,#15.3,
    DC_sbend_length: float = 150,
    DC_central_straight_length: float = 16.92,
    DC_coupl_wg_sep: float = 0.6,
    DC_coup_wg_width: float = 1.2,
    DC_ubend_sep: float = 100,
    h_racetrack: float = 200.0,
    ls: float = 1788, #3850
    rf_gap:float = 4,
    rf_central_conductor_width: float = 21.0,
    h: float = 3.0,
    r: float = 44.7,
    t: float = 7.0,
    s: float = 1.5,
    c: float = 5.0,
)-> gf.Component:

    lextra = 0
    #define subcomponents
    u_bend_racetrack = lnoi400.cells.U_bend_racetrack(
        v_offset = h_racetrack,
        p = 1.0,
        with_arc_floorplan = True,
        cross_section = RT_cross_section,
    )

    ecoup = mpl.asymmetric_directional_coupler_racetrack(
        cross_section_io=RT_cross_section,
        cross_section_bus=DC_cross_section,
        io_wg_sep = DC_io_wg_sep*2,
        sbend_length = DC_sbend_length,
        central_straight_length = DC_central_straight_length,
        coupl_wg_sep = DC_coupl_wg_sep,
        coup_wg_width = DC_coup_wg_width,
    )
    ec_xsize = ecoup.xsize

    dc_wg_sep = gf.components.straight(
        length = DC_ubend_sep,
        cross_section=RT_cross_section,
    )
    dc_wg_shorter = gf.components.straight(
        length = ls+2*lextra-ec_xsize-DC_ubend_sep,
        cross_section=RT_cross_section,
    )

    dc_wg = gf.components.straight(
        length = ls+2*lextra,
        cross_section=RT_cross_section,
    )


    def racetrack():
        # push subcomponents to design
        race_track = gf.Component()
        u_bend_ref1 = race_track << u_bend_racetrack
        u_bend_ref2 = race_track << u_bend_racetrack
        straight_ref1 = race_track << dc_wg_shorter
        straight_sep_ref = race_track << dc_wg_sep
        ec_top = race_track << ecoup
        straight_ref2 = race_track << dc_wg


        u_bend_ref1.connect("o2", straight_ref1.ports["o2"])
        straight_ref2.connect("o2", u_bend_ref1.ports["o1"])
        ec_top.connect("o3", straight_ref1["o1"])
        straight_sep_ref.connect("o1", ec_top.ports["o4"])
        u_bend_ref2.connect("o1", straight_sep_ref.ports["o2"])
        #u_bend_ref2.connect("o1", ec_top.ports["o4"])
        u_bend_ref2.connect("o2", straight_ref2.ports["o1"])

        exposed_ports = [
            ("o1", ec_top.ports["o1"]),
            ("o2", ec_top.ports["o2"])
        ]
        [race_track.add_port(name=name, port=port) for name, port in exposed_ports]

        return race_track

    # create assembly
    c = gf.Component()

    race_track_top = c << racetrack()
    race_track_bottom = c << racetrack()
    race_track_bottom.dmirror_y()
    race_track_bottom.dmovey(comb_sep)

    # Expose the ports
    exposed_ports = [
        ("ocb", race_track_top.ports["o1"]),
        ("o2", race_track_top.ports["o2"]),
        ("oct", race_track_bottom.ports["o1"]),
        ("o3", race_track_bottom.ports["o2"]),
    ]

    [c.add_port(name=name, port=port) for name, port in exposed_ports]

    mmi = c << lnoi400.cells.mmi1x2_optimized1550()
    mmi.drotate(90)
    mmi.dmovey(mmi.ports["o1"].dcenter[1], 0.5*c.ports["ocb"].dcenter[1]+0.5*c.ports["oct"].dcenter[1]-comb_sep/2.0-h_racetrack)
    mmi.dmovex(mmi.ports["o3"].dcenter[0], c.ports["ocb"].dcenter[0]-DC_ubend_sep-h_racetrack)

    routing_roc = 75.0
    routing_bend = partial(
        gf.components.bend_euler,
        #lnoi400.cells.S_bend_vert,
        radius=routing_roc,
        with_arc_floorplan=True,
    )

    gf.routing.route_single(
        c,
        port1=mmi.ports["o2"],
        port2=c.ports["oct"],
        cross_section="xs_rwg1000",
        bend = routing_bend,
        straight="straight_rwg1000",
    )
    gf.routing.route_single(
        c,
        port1=mmi.ports["o3"],
        port2=c.ports["ocb"],
        cross_section="xs_rwg1000",
        bend = routing_bend,
        straight="straight_rwg1000",
    )
    exposed_ports2 = [
        ("o1", mmi.ports["o1"]),
    ]
    [c.add_port(name=name, port=port) for name, port in exposed_ports2]

    return c


#####################
# Dual EO comb
#####################
@gf.cell
def dualEOcomb2(
    comb_sep: float = 101.25+50,
    RT_cross_section: CrossSectionSpec = "xs_rwg3000",
    DC_cross_section: CrossSectionSpec = "xs_rwg1000",
    DC_io_wg_sep: float = 25,#15.3,
    DC_sbend_length: float = 150,
    DC_central_straight_length: float = 16.92,
    DC_coupl_wg_sep: float = 0.6,
    DC_coup_wg_width: float = 1.2,
    DC_ubend_sep: float = 100,
    h_racetrack: float = 200.0,
    ls: float = 1788, #3850
    rf_gap:float = 4,
    rf_central_conductor_width: float = 21.0,
    h: float = 3.0,
    r: float = 44.7,
    t: float = 7.0,
    s: float = 1.5,
    c: float = 5.0,
    RF_res_layout_path = None,
    RF_pad_layout_path = None,
)-> gf.Component:

    # create assembly
    c = gf.Component()

    def singlecomb():
        return EOcomb(
                RT_cross_section = RT_cross_section,
                DC_cross_section = DC_cross_section,
                DC_io_wg_sep = DC_io_wg_sep,
                DC_sbend_length = DC_sbend_length,
                DC_central_straight_length = DC_central_straight_length,
                DC_coupl_wg_sep = DC_coupl_wg_sep,
                DC_coup_wg_width = DC_coup_wg_width,
                DC_ubend_sep = DC_ubend_sep,
                h_racetrack = h_racetrack,
                ls = ls,
                rf_gap = rf_gap,
                rf_central_conductor_width = rf_central_conductor_width,
                h = h,
                r = r,
                t = t,
                s = s,
                c = c,
                RF_res_layout_path = RF_res_layout_path,
                RF_pad_layout_path = RF_pad_layout_path,
        )
    race_track_top = c << singlecomb()
    race_track_top.dmirror_y()
    race_track_bottom = c << singlecomb()
    #race_track_bottom.dmirror_y()
    race_track_bottom.dmovey(comb_sep)

    # Expose the ports
    exposed_ports = [
        ("ocb", race_track_top.ports["o1"]),
        ("o2", race_track_top.ports["o2"]),
        ("oct", race_track_bottom.ports["o1"]),
        ("o3", race_track_bottom.ports["o2"]),
    ]

    [c.add_port(name=name, port=port) for name, port in exposed_ports]

    mmi = c << lnoi400.cells.mmi1x2_optimized1550()
    mmi.drotate(90)
    mmi.dmovey(mmi.ports["o1"].dcenter[1], 0.5*c.ports["ocb"].dcenter[1]+0.5*c.ports["oct"].dcenter[1]-comb_sep/2.0-h_racetrack)
    mmi.dmovex(mmi.ports["o3"].dcenter[0], c.ports["ocb"].dcenter[0]-DC_ubend_sep-h_racetrack)

    routing_roc = 75.0
    routing_bend = partial(
        gf.components.bend_euler,
        #lnoi400.cells.S_bend_vert,
        radius=routing_roc,
        with_arc_floorplan=True,
    )

    gf.routing.route_single(
        c,
        port1=mmi.ports["o2"],
        port2=c.ports["oct"],
        cross_section="xs_rwg1000",
        bend = routing_bend,
        straight="straight_rwg1000",
    )
    gf.routing.route_single(
        c,
        port1=mmi.ports["o3"],
        port2=c.ports["ocb"],
        cross_section="xs_rwg1000",
        bend = routing_bend,
        straight="straight_rwg1000",
    )
    exposed_ports2 = [
        ("o1", mmi.ports["o1"]),
    ]
    [c.add_port(name=name, port=port) for name, port in exposed_ports2]

    return c



#####################
# Dual EO comb with horizontal mmi input and sbend transition
#####################
@gf.cell
def dualEOcomb_sbend(
    comb_sep: float = 101.25+50,
    RT_cross_section: CrossSectionSpec = "xs_rwg3000",
    DC_cross_section: CrossSectionSpec = "xs_rwg1000",
    DC_io_wg_sep: float = 25,#15.3,
    DC_sbend_length: float = 150,
    DC_central_straight_length: float = 16.92,
    DC_coupl_wg_sep: float = 0.6,
    DC_coup_wg_width: float = 1.2,
    DC_ubend_sep: float = 100,
    h_racetrack: float = 200.0,
    ls: float = 1788, #3850
    rf_gap:float = 4,
    rf_central_conductor_width: float = 21.0,
    h: float = 3.0,
    r: float = 44.7,
    t: float = 7.0,
    s: float = 1.5,
    c: float = 5.0,
    RF_res_layout_path = None,
    RF_pad_layout_path = None,
)-> gf.Component:

    # create assembly
    c = gf.Component()

    def singlecomb():
        return EOcomb(
                RT_cross_section = RT_cross_section,
                DC_cross_section = DC_cross_section,
                DC_io_wg_sep = DC_io_wg_sep,
                DC_sbend_length = DC_sbend_length,
                DC_central_straight_length = DC_central_straight_length,
                DC_coupl_wg_sep = DC_coupl_wg_sep,
                DC_coup_wg_width = DC_coup_wg_width,
                DC_ubend_sep = DC_ubend_sep,
                h_racetrack = h_racetrack,
                ls = ls,
                rf_gap = rf_gap,
                rf_central_conductor_width = rf_central_conductor_width,
                h = h,
                r = r,
                t = t,
                s = s,
                c = c,
                RF_res_layout_path = RF_res_layout_path,
                RF_pad_layout_path = RF_pad_layout_path,
        )
    race_track_top = c << singlecomb()
    race_track_top.dmirror_y()
    race_track_bottom = c << singlecomb()

    race_track_bottom.dmovey(comb_sep)

    # Expose the ports
    exposed_ports = [
        ("ocb", race_track_top.ports["o1"]),
        ("o2", race_track_top.ports["o2"]),
        ("oct", race_track_bottom.ports["o1"]),
        ("o3", race_track_bottom.ports["o2"]),
    ]

    [c.add_port(name=name, port=port) for name, port in exposed_ports]

    mmi = c << lnoi400.cells.mmi1x2_optimized1550()

    mmi.dmovey(mmi.ports["o1"].dcenter[1], 0.5*c.ports["ocb"].dcenter[1]+0.5*c.ports["oct"].dcenter[1])
    mmi.dmovex(mmi.ports["o1"].dcenter[0], c.ports["ocb"].dcenter[0]-DC_ubend_sep-h_racetrack)


    sbend_dx_straight = 5.0
    sbend_dy = c.ports["oct"].dcenter[1] - mmi.ports["o2"].dcenter[1]
    sbend_dx = c.ports["oct"].dcenter[0] - mmi.ports["o2"].dcenter[0]-2*sbend_dx_straight
    sbend_top = lnoi400.cells.S_bend_vert(v_offset=sbend_dy,
                                        h_extent = sbend_dx,
                                        dx_straight=sbend_dx_straight,
                                        cross_section=DC_cross_section
                                        )
    sbend_bot = lnoi400.cells.S_bend_vert(v_offset=sbend_dy,
                                        h_extent = sbend_dx,
                                        dx_straight=sbend_dx_straight,
                                        cross_section=DC_cross_section
                                        )
    
    sbend_ref_top = c << sbend_top
    sbend_ref_bot = c << sbend_bot
    sbend_ref_bot.dmirror_y()
    sbend_ref_top.connect("o2", mmi.ports["o2"])
    sbend_ref_bot.connect("o2", mmi.ports["o3"])

    exposed_ports2 = [
        ("o1", mmi.ports["o1"]),
    ]
    [c.add_port(name=name, port=port) for name, port in exposed_ports2]

    return c


#####################
# 30GHz FSR Racetrack Resonator (Final Design)
#####################
@gf.cell
def racetrack_30GHzFSR():

    #define constants for race track
    h_racetrack = 200.0
    length = 1800.0

    #define subcomponents
    u_bend_racetrack = lnoi400.cells.U_bend_racetrack(
        v_offset = h_racetrack,
        p = 1.0,
        with_arc_floorplan = True,
        cross_section = "xs_rwg3000",
    )

    #length_check = u_bend_racetrack.info['length']
    #print(length_check)   

    dc_wg = gf.components.straight(
        length = length,
        cross_section="xs_rwg3000",
    )

    def racetrack():
        # push subcomponents to design
        race_track = gf.Component()
        u_bend_ref1 = race_track << u_bend_racetrack
        u_bend_ref2 = race_track << u_bend_racetrack
        straight_ref1 = race_track << dc_wg
        straight_ref2 = race_track << dc_wg

        u_bend_ref1.connect("o2", straight_ref1.ports["o2"])
        straight_ref2.connect("o2", u_bend_ref1.ports["o1"])
        u_bend_ref2.connect("o1", straight_ref1.ports["o1"])
        u_bend_ref2.connect("o2", straight_ref2.ports["o1"])

        #race_track.flatten()

        return race_track

    # create assembly
    c = gf.Component()
    c << racetrack()

    return c

#####################
# dOR_EOM_DC with EMMI splitters
#####################
@gf.cell
def dOR_EOM_DC_EMMI(
    RT_cross_section: CrossSectionSpec = "xs_rwg3000",
    DC_cross_section: CrossSectionSpec = "xs_rwg1000",
    DC_io_wg_sep: float = 25,#15.3,
    DC_sbend_length: float = 150,
    DC_central_straight_length: float = 16.92,
    DC_coupl_wg_sep: float = 0.6,
    DC_coup_wg_width: float = 1.2,
    DC_ubend_sep: float = 100,
    h_racetrack: float = 200.0,
    ls: float = 1408.0, #3850
    lextra: float = 364/2, #205/2
    rf_gap: float = 4,
    rf_central_conductor_width: float = 21.0,
    h: float = 3.0,
    r: float = 44.7,
    t: float = 7.0,
    s: float = 1.5,
    c: float = 5.0,
    with_heater: bool = True,
    bias_tuning_section_length: float = 500.0,
    heater_offset: float = 1.5,
    heater_x: float = 0.0,
    heater_width: float = 1.0,
    heater_pad_size: tuple[float, float] = (75.0, 75.0),
    heater_xdisp: float = 0.0,
    start_width: float = 80.0,
    length_straight: float = 10.0,
    length_tapered: float = 190.0,
)-> gf.Component:

    circuit = gf.Component()

    # adding subcomponents
    splitter1 = lnoi400.cells.mmi1x2_optimized1550()
    splitter2 = lnoi400.cells.mmi2x2_optimized1550()

    EOM = dOR_EOM_DC(
        RT_cross_section = RT_cross_section,
        DC_cross_section = DC_cross_section,
        DC_io_wg_sep = DC_io_wg_sep,
        DC_sbend_length = DC_sbend_length,
        DC_central_straight_length = DC_central_straight_length,
        DC_coupl_wg_sep = DC_coupl_wg_sep,
        DC_coup_wg_width = DC_coup_wg_width,
        DC_ubend_sep = DC_ubend_sep,
        h_racetrack = h_racetrack,
        ls = ls, #3850
        lextra = lextra, #205
        rf_gap = rf_gap,
        rf_central_conductor_width = rf_central_conductor_width,
        h = h,
        r = r,
        t = t,
        s = s,
        c = c,
        start_width = start_width,
        length_straight = length_straight, 
        length_tapered = length_tapered, 
    )

    if with_heater:
        heater = lnoi400.cells.heater_straight_single(
            length=bias_tuning_section_length,
            width=heater_width,
            # offset=heater_offset,
            pad_size=heater_pad_size,
        )

        heater_disp = [
            #heater_xdisp + ls + lextra + h_racetrack - heater.xsize/2 + 0*splitter2.xsize/2,
            heater_xdisp + ls/2 + lextra + h_racetrack,
            rf_central_conductor_width/2 + rf_gap/2 + h + s + h_racetrack + DC_io_wg_sep + heater_offset + heater_width/2
            ]
        
        heater_disp2 = [
            0.0,
            rf_central_conductor_width/2 + rf_gap/2 + h + s + h_racetrack + heater_offset + heater_width/2  + 0.2
        ]

    # Push to circuit
    splitter1_ref = circuit << splitter1
    splitter2_ref = circuit << splitter2
    EOM_ref = circuit << EOM
    heater_ref = circuit << heater if with_heater else None
    heater_ref2 = circuit << heater if with_heater else None
    heater_ref3 = circuit << heater if with_heater else None


    usafetygap = 0.4*h_racetrack
    # Move
    splitter1_ref.dmovex(splitter1_ref.ports["o1"].dcenter[0], EOM.dx - lextra -ls/2.0 - usafetygap - 1.0*h_racetrack - splitter1.xsize)
    splitter2_ref.rotate(180)
    splitter2_ref.dmovex(splitter2_ref.ports["o2"].dcenter[0],EOM.dx + lextra +ls/2.0 + usafetygap + 1.0*h_racetrack + splitter2.xsize)


    heater_ref.dmovex(heater_ref.ports["e2"].dcenter[0], EOM.dx + heater_disp[0]) if with_heater else None
    heater_ref.movey(heater_disp[1]) if with_heater else None

    heater_ref2.movey(heater_disp2[1]) if with_heater else None
    heater_ref3.movey(heater_disp2[1]) if with_heater else None

    heater_ref2.dmovex(heater_ref2.ports["e2"].dcenter[0],  heater_x + EOM.dxsize/2 - heater_ref2.dxsize/2) if with_heater else None
    heater_ref3.dmovex(heater_ref3.ports["e2"].dcenter[0],  heater_x + EOM.dxsize/2 - heater_ref3.dxsize/2) if with_heater else None


    heater_ref3.mirror_y() if with_heater else None
    

    # Connect
    routing_bend = partial(
        gf.components.bend_euler,
        radius=75.0,
        with_arc_floorplan=True,
    )

    # Routing call gives a warning
    gf.routing.route_single(
        circuit,
        splitter1_ref.ports["o2"],
        EOM.ports["o1"],
        start_straight_length=5.0,
        end_straight_length=5.0,
        cross_section="xs_rwg1000",
        bend=routing_bend,
        straight="straight_rwg1000",
    )

    gf.routing.route_single(
        circuit,
        splitter1_ref.ports["o3"],
        EOM.ports["o3"],
        start_straight_length=5.0,
        end_straight_length=5.0,
        cross_section="xs_rwg1000",
        bend=routing_bend,
        straight="straight_rwg1000",
    )

    gf.routing.route_single(
        circuit,
        splitter2_ref.ports["o4"],
        EOM.ports["o2"],
        start_straight_length=5.0,
        end_straight_length=5.0,
        cross_section="xs_rwg1000",
        bend=routing_bend,
        straight="straight_rwg1000",
    )

    gf.routing.route_single(
        circuit,
        splitter2_ref.ports["o3"],
        EOM.ports["o4"],
        start_straight_length=5.0,
        end_straight_length=5.0,
        cross_section="xs_rwg1000",
        bend=routing_bend,
        straight="straight_rwg1000",
    )

    # Define ports
    exposed_ports = []

    exposed_ports += [
        ("e1", EOM_ref.ports["e1"]),
        ("e2", EOM_ref.ports["e2"]),
    ]

    if with_heater:
        exposed_ports += [
            ("e3", heater_ref.ports["e1"]),
            ("e4", heater_ref.ports["e2"],),
            ("e5", heater_ref2.ports["e1"],),
            ("e6", heater_ref2.ports["e2"],),
            ("e7", heater_ref3.ports["e1"],),
            ("e8", heater_ref3.ports["e2"],),
        ]


    exposed_ports += [
        ("o1", splitter1_ref.ports["o1"]),
        ("o2", splitter2_ref.ports["o1"]),
        ("o3", splitter2_ref.ports["o2"]),
    ]

    [circuit.add_port(name=name, port=port) for name, port in exposed_ports]

    return circuit