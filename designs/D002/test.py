"""
Microring resonator with rectangular notch grating function.
@author: Andrea Caruso; andrea.caruso@tuni.fi

Microring resonator-based vortex beam emitter function.  
"""
from __future__ import annotations

import gdsfactory as gf
import math

from gdsfactory.typings import CrossSectionSpec

#from typing import Union, Any, TypeVar, cast

#WG_CROSS_SECTION: CrossSectionSpec = "xs_rwg800"
WG_CROSS_SECTION: CrossSectionSpec = "strip"
# Above crossection should be changed when there is access to lnoi400.



@gf.cell
def ring_vortex_beam_emitter(
    q: int = 338,
    R_ring: float = 50,
    W_wg: float = 0.8,
    W_gap: float = 0.3,
    W_notch: float = 0.25,
    notch_type: str = "S", 
    resolution: int = 0.1, 
    show_ports: bool = False, 
    layer: tuple[int, int] = (2, 0),
    cross_section: CrossSectionSpec = WG_CROSS_SECTION
) -> gf.Component:
    
    W_margin = 0.25 * W_notch
    rad_step = 2*math.pi/q
    
    c = gf.Component()
    
    ring = c.create_inst(gf.components.ring(radius=R_ring, 
                                            width=W_wg, 
                                            angle_resolution=resolution,
                                            layer=layer))
    
    for i in range(q):
            if notch_type == "S":
                ni = c.create_inst(gf.components.taper(length=W_notch+W_margin, 
                                                       width1=W_notch, 
                                                       width2=W_notch,
                                                       layer=layer))
                
                R_i = R_ring - W_notch - W_wg/2
                
                ni.rotate(i * rad_step * 180/math.pi)
                ni.move((R_i * math.cos(i * rad_step), R_i * math.sin(i * rad_step)))
            elif notch_type == "C_in":
                ni = c << gf.components.circle(radius=W_notch/2,
                                               angle_resolution=2.5, 
                                               layer=(layer[0], 1))
                
                R_i = R_ring - W_notch/2 - 0.3 - W_wg/2
                
                ni.rotate(i * rad_step * 180/math.pi)
                ni.move((R_i * math.cos(i * rad_step), R_i * math.sin(i * rad_step)))
        
    bus = c.create_inst(gf.components.rectangle(size=(2*R_ring+W_wg, W_wg), 
                                                layer=layer))
    
    bus.move((-R_ring - W_wg/2, -R_ring - 3*W_wg/2 - W_gap))
    
    c.flatten()
    
    c.add_port(name="o1", 
               center=(R_ring+W_wg/2, -R_ring-W_wg-W_gap), 
               width=W_wg, 
               orientation=0, 
               layer=layer)
    
    c.add_port(name="o2", 
               center=(-R_ring-W_wg/2, -R_ring-W_wg-W_gap), 
               width=W_wg,
               orientation=180, 
               layer=layer)
    
    c.rotate(180)
    
    c.movey(-W_wg - W_gap - R_ring)
    
    if show_ports == True:
        c.draw_ports()
    
    
    return c