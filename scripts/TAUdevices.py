# Devices by TAU

#####################################################################################
# Authors: Ajwaad Quashef, ORC 2025

import gdsfactory as gf
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec, ComponentSpec
import numpy as np
import components as orc_components

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
from lnoi400.tech import LAYER, xs_uni_cpw
from lnoi400.cells import uni_cpw_straight, S_bend_vert, eo_phase_shifter_no_taper, L_turn_bend
from gdsfactory.routing import route_single_sbend
from gdsfactory.routing import route_quad
from lnoi400.spline import (
    bend_S_spline,
    bend_S_spline_varying_width,
    spline_clamped_path,
)


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
    edge_coupler = orc_components.linear_inverse_taper_AQ(
                                input_ext=input_ext,
                                )
    st_wg = orc_components.straight_rwg2000(
        length = 210
    )
    phase_modulator = orc_components.eo_phase_modulator_AQ(modulation_length=4500.0)


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
    splitter = orc_components.custom_mmi_AQ()
    mirror = orc_components.loop_mirror_AQ(splitter='custom_mmi_AQ', cross_section='xs_rwg2000')
    mzm = orc_components.mzm_custom_AQ(modulation_length=4000)

    input_ext = 10.0
    edge_coupler = orc_components.linear_inverse_taper_AQ(
                                input_ext=input_ext,
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
    mmi_component = orc_components.custom_mmi()
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




    # 7. Generate the waveguide route
    route1 = gf.routing.route_single(c,
        port1=coarse_output_mmi,
        port2=medium_output_mmi,
        cross_section='xs_rwg2000',
        bend='L_turn_bend',
        radius= 100,
        straight='straight_rwg2000'
    )


    #8 add phase shifter


    # Create the phase shifter component
    phase_shifter = pdk.cells.eo_phase_shifter(modulation_length= 2500)
    ps_ref = c.add_ref(phase_shifter)

    ps_ref.movey(-500)
    ps_ref.movex(-200)

    return c

##################################################################################### End: Redwan Islam