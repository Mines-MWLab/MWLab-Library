# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:41:43 2025

@author: ccanca
"""
from __future__ import annotations

import gdsfactory as gf
import math
import numpy as np
import numpy.typing as npt

from gdsfactory.path import extrude_transition, spiral_archimedean, transition
from gdsfactory.typings import CrossSectionSpec

from scipy.integrate import quad
from scipy.optimize import brentq

from typing import Union, Any, TypeVar, cast

from ring_vortex_beam_emitter import ring_vortex_beam_emitter
from spiral_vortex_beam_emitter_equal_arc_spacing import spiral_vortex_beam_emitter_equal_arc_spacing
from euler_curve import euler_curve
from input_coupler import input_coupler
#from nanophotonics_chip import nanophotonics_chip

#@gf.cell
def nanophotonics_chip():
    
    def input_coupler(
        W_wg: float = 0.8,
        L_wg: float = 1000,
        layer: tuple = (2, 0)
    ) -> gf.Component:
        """Returns a straight waveguide with no ports, just a placeholder for
           an input coupler/waveguide.

        Args:
            W_wg: width of the waveguide.
            L_wg: length of the waveguide.
            layer: layer at which to position component.
        """

        c = gf.Component()
        
        wg = c << gf.components.taper(length=L_wg, 
                                      width1=W_wg, 
                                      width2=W_wg,
                                      layer=layer)
        
        c.add_port(name="o1",
                   center=(L_wg, 0),
                   width=W_wg,
                   orientation=0,
                   layer=layer)
        
        c.add_port(name="o2",
                   center=(0, 0),
                   width=W_wg,
                   orientation=180,
                   layer=layer)
        
        #c.draw_ports()
        
        return c

    # Used chip size constants.
    W = 10000/4
    H = 5000
    
    c = gf.Component()
    
    # Defining the chip size and an area for it for reference
    #  (could have also added a rectangle)
    #testsp = c << spiral_vortex_beam_emitter_equal_arc_spacing(notch_type="C_out")
    
    base = c << gf.components.rectangle(size=(W, H), layer=(0, 0))
    
    # First block.
    for i in range(1, 16):
        H = 5000
        in_i = c << input_coupler(L_wg=i*125)
        in_i.movey(H - 125 - i*25)
    
        if i in range(1, 6):
            N = [331, 332, 333, 334, 335]

            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.25, 
                                                                     q=N[i-1],
                                                                     notch_type="S")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-74.4, H - 74 - i*25))
    
        if i in range(6, 11):
            N = [331, 332, 333, 334, 335]
            
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.25, 
                                                                     q=N[i-5-1], 
                                                                     notch_type="C_in")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-74.4, H - 74 - i*25))
            
        if i in range(11, 16):
            N = [331, 332, 333, 334, 335]
            
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.25, 
                                                                     q=N[i-10-1], 
                                                                     notch_type="C_out")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-74.4, H - 74 - i*25))
    
    # Second block.
    for i in range(1, 16):
        H = 4174.2
        in_i = c << input_coupler(L_wg=i*125)
        in_i.movey(H - 125 - i*25)
    
        if i in range(1, 6):
            N = [331, 332, 333, 334, 335]

            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.3,
                                                                     q=N[i-1], 
                                                                     variable_pillar_dist=True,
                                                                     notch_type="S")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-74.4, H - 74 - i*25))
    
        if i in range(6, 11):
            N = [331, 332, 333, 334, 335]

            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.3,
                                                                     q=N[i-5-1],
                                                                     variable_pillar_dist=True,
                                                                     notch_type="C_in")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-74.4, H - 74 - i*25))
            
        if i in range(11, 16):
            N = [331, 332, 333, 334, 335]

            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.3, 
                                                                     q=N[i-10-1],
                                                                     variable_pillar_dist=True,
                                                                     notch_type="C_out")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-74.4, H - 74 - i*25))
    
    # Third block.
    for i in range(1, 16):
        H = 3348.4
        in_i = c << input_coupler(L_wg=i*125)
        in_i.movey(H - 125 - i*25)
    
        if i in range(1, 6):
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.25, 
                                                                     loops_spiral=2, 
                                                                     q=649, 
                                                                     notch_type="S")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-71.4, H - 71 - i*25))
    
        if i in range(6, 11):
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.25, 
                                                                     loops_spiral=2, 
                                                                     q=649, 
                                                                     notch_type="C_in")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-71.4, H - 71 - i*25))
            
        if i in range(11, 16):
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.3, 
                                                                     loops_spiral=2, 
                                                                     q=649, 
                                                                     notch_type="C_out")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-71.4, H - 71 - i*25))
    
    # Fourth block.
    for i in range(1, 16):
        H = 2522.6
        in_i = c << input_coupler(L_wg=i*125)
        in_i.movey(H - 125 - i*25)
    
        if i in range(1, 6):
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.25, 
                                                                     loops_spiral=3, 
                                                                     q=953, 
                                                                     notch_type="S")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-68.4, H - 68 - i*25))
    
        if i in range(6, 11):
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.25, 
                                                                     loops_spiral=3, 
                                                                     q=953, 
                                                                     notch_type="C_in")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-68.4, H - 68 - i*25))
            
        if i in range(11, 16):
            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(W_notch=0.3, 
                                                                     loops_spiral=3, 
                                                                     q=953, 
                                                                     notch_type="C_out")
            sp_i.rotate(270)
            sp_i.move(((i+1)*125-68.4, H - 68 - i*25)) 
    
    
        for i in range(31, 40):
            N   = [336, 338, 340, 336, 338, 340, 336, 338, 340]
            gap = [0.30, 0.35, 0.40, 0.30, 0.35, 0.40, 0.30, 0.35, 0.40]

            sp_i = c << ring_vortex_beam_emitter(q=N[i-30-1], 
                                                 W_gap=gap[i-30-1], 
                                                 notch_type="C_in")
            sp_i.rotate(180)
            sp_i.move((1000+50+0.4, H-i*125))

            euler_i = c << euler_curve()
            euler_i.move((1000+150.8+(8-(i-30-1))*125, H-i*125-50))
            
            in1_i = c << input_coupler(L_wg=(8-(i-30-1))*125)
            in1_i.move((1000+100.8, H-i*125))
            
            in2_i = c << input_coupler(L_wg=1075-(i-30-1)*125)
            in2_i.rotate(90)
            in2_i.movex(2150.8-(i-30-1)*125)
    
    
    #c.plot()
    #c.show()

    return c
