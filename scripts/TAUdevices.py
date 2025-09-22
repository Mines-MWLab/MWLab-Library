# Devices by TAU

#####################################################################################
# Authors: Ajwaad Quashef, ORC 2025

import gdsfactory as gf
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec, ComponentSpec
import numpy as np
from . import components as orc_components

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
from lnoi400.tech import LAYER, xs_uni_cpw


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
