

from __future__ import annotations

import gdsfactory as gf

from gdsfactory.typings import CrossSectionSpec

#WG_CROSS_SECTION: CrossSectionSpec = "xs_rwg800"
WG_CROSS_SECTION: CrossSectionSpec = "strip"
# Above crossection should be changed when there is access to lnoi400.


@gf.cell
def grating_structure(
    W_structure: float = 100, 
    W_element: float = 0.7, 
    period: float = 0.911, 
    grating_lines: int = 110, 
    layer: tuple[int, int] = (2, 0), 
    cross_section: CrossSectionSpec = WG_CROSS_SECTION
) -> gf.Component:
    
    def grating_line(
        W_wg: float = 0.7,
        L_wg: float = 100,
        layer: tuple = layer
    ) -> gf.Component:
        
        c = gf.Component()
        
        _ = c << gf.components.taper(length=L_wg, 
                                     width1=W_wg,
                                     width2=W_wg, 
                                     cross_section=cross_section, 
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
    
    c = gf.Component()
    
    for i in range(1, grating_lines+1):
        wgi = c << grating_line(W_element, W_structure)
        wgi.movey((i-1)*period)
        
    c.movey(W_element/2)
    
    return c