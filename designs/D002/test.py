# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:41:43 2025

@author: ccanca
"""
from __future__ import annotations

import gdsfactory as gf
import lnoi400
import math
import numpy as np
import numpy.typing as npt

from gdsfactory.path import extrude_transition, spiral_archimedean, transition, arc
from gdsfactory.typings import CrossSectionSpec

from scipy.integrate import quad
from scipy.optimize import brentq


from typing import Union, Any, TypeVar, cast


WG_CROSS_SECTION: CrossSectionSpec = "xs_rwg800"


@gf.cell
def nanophotonics_chip():
    
    def euler_curve(
        R_curve: float = 50,
        W_curve: float = 0.8,
        angle: int = 90,
        npoints: int = 1000,
        cross_section: CrossSectionSpec = WG_CROSS_SECTION,
    ) -> gf.Component:
        c = gf.Component()

        euler = c << gf.components.bend_euler(
            radius=R_curve,
            width=W_curve,
            angle=angle,
            npoints=npoints,
            cross_section=cross_section,
        )

        c.rotate(90)

        c.add_port("o1", port=euler.ports["o1"])
        c.add_port("o2", port=euler.ports["o2"])

        return c

    def input_coupler(
        W_wg: float = 0.8,
        L_wg: float = 1000,
        cross_section: CrossSectionSpec = WG_CROSS_SECTION,
    ) -> gf.Component:

        c = gf.Component()
        wg = c << gf.components.taper(
            length=L_wg,
            width1=W_wg,
            width2=W_wg,
            cross_section=cross_section,
        )

        c.add_port("o1", port=wg.ports["o1"])
        c.add_port("o2", port=wg.ports["o2"])

        return c

    def ring_vortex_beam_emitter(
        q: int = 338,
        R_ring: float = 50,
        W_wg: float = 0.8,
        W_gap: float = 0.3,
        W_notch: float = 0.25,
        notch_type: str = "S", 
        resolution: int = 0.1,
        show_ports: bool = False,
        cross_section: CrossSectionSpec = WG_CROSS_SECTION,
    ) -> gf.Component:
        W_margin = 0.25 * W_notch
        rad_step = 2*math.pi/q
        
        c = gf.Component()
        
        ring_path = arc(radius=R_ring, angle=360, npoints=int(max(360, resolution * 360)))
        ring_component = gf.path.extrude(ring_path, gf.get_cross_section(cross_section, width=W_wg))
        ring = c << ring_component
        ring.dmovey(-R_ring)
        
        for i in range(q):
                if notch_type == "S":
                    ni = c << gf.components.taper(
                        length=W_notch + W_margin,
                        width1=W_notch,
                        width2=W_notch,
                        cross_section=cross_section,
                    )
                    
                    R_i = R_ring - W_notch - W_wg/2
                    
                    ni.rotate(i * rad_step * 180/math.pi)
                    ni.move((R_i * math.cos(i * rad_step), R_i * math.sin(i * rad_step)))
                
                elif notch_type == "C_in":
                    ni = c << gf.components.circle(radius=W_notch/2,
                                                   angle_resolution=2.5, 
                                                   layer=(2, 1))
                    
                    R_i = R_ring - W_notch/2 - 0.3 - W_wg/2
                    
                    ni.rotate(i * rad_step * 180/math.pi)
                    ni.move((R_i * math.cos(i * rad_step), R_i * math.sin(i * rad_step)))
            
        bus = c << gf.components.straight(
            length=2 * R_ring + W_wg,
            width=W_wg,
            cross_section=cross_section,
        )

        bus.movex(-R_ring - W_wg / 2)
        bus.movey(-R_ring - 3 * W_wg / 2 - W_gap)
        
        c.flatten()
        
        c.add_port(
            name="o1",
            center=(R_ring + W_wg / 2, -R_ring - W_wg - W_gap),
            width=W_wg,
            orientation=0,
            layer=(2, 0),
        )

        c.add_port(
            name="o2",
            center=(-R_ring - W_wg / 2, -R_ring - W_wg - W_gap),
            width=W_wg,
            orientation=180,
            layer=(2, 0),
        )
        
        c.rotate(180)
        
        c.movey(-1.5*W_wg - W_gap - R_ring)
        
        if show_ports == True:
            c.draw_ports()
        
        
        return c

    def taper_spiral_waveguide(
        separation: float = 3.0,
        width_in: float = 0.2,
        width_tip: float = 0.2,
        number_of_loops: float = 1,
        npoints: int = 1000,
        min_bend_radius: float = 6.0,
        taper: str = "linear",
        cross_section: CrossSectionSpec = WG_CROSS_SECTION,
    ) -> gf.Component:
        """Helper function; returns doped taper to terminate waveguides.

        Args:
            separation: separation between the loops.
            width_in: width of the default cross-section at the input of the termination.
            width_tip: width of the default cross-section at the end of the termination.
            number_of_loops: number of loops in the spiral.
            npoints: points for the spiral.
            min_bend_radius: minimum bend radius for the spiral.
            taper: type of width change function.
            cross_section: input cross-section.
        """
        cross_section_in = gf.get_cross_section(cross_section, width=width_in)
        cross_section_tip = gf.get_cross_section(cross_section, width=width_tip)
        
        xs = transition(
            cross_section1=cross_section_tip, 
            cross_section2=cross_section_in, 
            width_type=taper,
        )
        
        #min_bend_radius = min_bend_radius or cross_section_in.radius_min
        #assert min_bend_radius
        
        path = spiral_archimedean(
            min_bend_radius=min_bend_radius,
            separation=separation/2,
            number_of_loops=number_of_loops,
            npoints=npoints,
        )
        path.start_angle = 0
        path.end_angle   = 0
        
        spiral = extrude_transition(path, transition=xs)
        c = gf.Component()
        ref = c << spiral
        
        return c

    def spiral_length(a, r0, theta1, theta2):
        """Arc length of Archimedean spiral between theta1 and theta2."""
        def integrand(theta):
            r = a * theta + r0
            dr_dtheta = a
            return np.sqrt(dr_dtheta**2 + r**2)
        
        length, _ = quad(integrand, theta1, theta2)
        return length

    def equal_arc_spiral(a, r0, theta_max, N):
        """Divide Archimedean spiral into N equal arc-length sections."""
        L_total = spiral_length(a, r0, 0, theta_max)
        L_target = np.linspace(0, L_total, N+1)  # target cumulative lengths
        
        thetas = [0.0]
        for i in range(1, N+1):
            def f(theta):
                return spiral_length(a, r0, 0, theta) - L_target[i]
            theta_i = brentq(f, thetas[-1], theta_max)
            thetas.append(theta_i)
        
        radii = [a*t + r0 for t in thetas]
        return np.array(thetas), np.array(radii), L_total

    def spiral_vortex_beam_emitter_equal_arc_spacing(
        q: int = 331,
        W_spiral: float = 0.8,
        W_notch: float = 0.25,
        notch_type: str = "S",
        loops_spiral: int = 1,
        loops_tail: int = 1,
        R_sep: float = 2,
        R_min: float = 48,
        direction: str = 'R',
        resolution: int = 1000,
        show_ports: bool = False,
        cross_section: CrossSectionSpec = WG_CROSS_SECTION,
    ) -> gf.Component:
        """Returns Archimedean spiral vortex beam emitter.
           All measurements with units given in [um] units.

        Args:
            q: number of notches per loop.
            W_in: width of spiral at input port.
            W_out: width of spiral at start of tail.
            W_notch: Width of the notches.
            L_notch: length of the notches.
            loops_spiral: number of loops that make up the spiral.
            loops_tail: number of loops that make up the tail.
            R_sep: separation between the loops.
            R_min: minimum loop radius.
            direction: the rotation direction of the spiral.
            taper: type of width change function; either "linear", or a list of 
                   polynomial fit parameters up to a 6th-degree polynomial fit.
                   <<<CURRENTLY WIP>>>.
            resolution: points for the spiral.
            layer: the layer in which to position the spiral.
            show_ports: determines whether ports are displayed.
        """
        
        L_in = R_min + loops_spiral * R_sep - W_spiral/2
        
        c = gf.Component()
        
        sp = c << taper_spiral_waveguide(
            separation=R_sep,
            width_in=W_spiral,
            width_tip=W_spiral,
            number_of_loops=loops_spiral,
            min_bend_radius=R_min,
            npoints=resolution,
            cross_section=cross_section,
        )
            
        sp.rotate(270)
            
        a         = -R_sep/(2*np.pi)     # [um/2pi] Growth rate of the spiral
        r0        = R_min + R_sep        # [um] Starting radius
        theta_max = loops_spiral*2*np.pi
        N         = q
        
        thetas, radii, L_total = equal_arc_spiral(a, r0, theta_max, N)
            
        for j in range(loops_spiral):
            
            for i in range(len(thetas)):
                R_at_spot = radii[i]
                
                if notch_type == "S":
                    ni = c << gf.components.taper(
                        length=W_notch + 0.25 * W_notch,
                        width1=W_notch,
                        width2=W_notch,
                        cross_section=cross_section,
                    )
                    
                    R_xy = (R_at_spot) + (0.25*W_notch) - (W_spiral/2)
                    
                    ni.rotate(180/np.pi * thetas[i] - 180)
                    ni.movex(R_xy * np.cos(thetas[i]))
                    ni.movey(R_xy * np.sin(thetas[i]))   
                    
                elif notch_type == "C_in":
                    ni = c << gf.components.circle(radius=W_notch/2,
                                                   angle_resolution=1/resolution, 
                                                   layer=(2, 1))
                    
                    R_xy = (R_at_spot) - W_notch/2 - 0.3 - (W_spiral/2)
                    
                    ni.movex(R_xy * np.cos(thetas[i]))
                    ni.movey(R_xy * np.sin(thetas[i]))   
                    
                elif notch_type == "C_out":
                    ni = c << gf.components.circle(radius=W_notch/2,
                                                   angle_resolution=1/resolution, 
                                                   layer=(2, 1))
                    
                    R_xy = (R_at_spot) + W_notch/2 + 0.3 + (W_spiral/2)
                    
                    ni.movex(R_xy * np.cos(thetas[i]))
                    ni.movey(R_xy * np.sin(thetas[i]))   
                    
        tail = c << taper_spiral_waveguide(
            separation=5 * R_sep,
            width_in=W_spiral,
            width_tip=0.25,
            number_of_loops=loops_tail,
            min_bend_radius=R_min - loops_tail * 5 * R_sep,
            taper="linear",
            npoints=resolution,
            cross_section=cross_section,
        )
            
        tail.rotate(270)
        
        c.rotate(1/q * 360)
        
        R_max = R_min + loops_spiral * R_sep
        
        # create_vinst() is needed for "all_angle components", otherwise 
        # create_inst() is equivalent to c << .
        arc = c.create_vinst(
            gf.components.bend_euler_all_angle(
                radius=R_max,
                width=W_spiral,
                angle=1 / q * 360,
                npoints=int(resolution * 1 / q),
                cross_section=cross_section,
            )
        )
        
        arc.rotate(90)
        arc.movex(R_max)
        
        straight = c << gf.components.taper(
            length=L_in,
            width1=W_spiral,
            width2=W_spiral,
            cross_section=cross_section,
        )
        straight.rotate(90)
        straight.movex(R_max)
        straight.movey(-L_in)
        
        c.flatten()
        
        if direction == 'L':
            c.mirror_x()
            
            c.add_port(name="o1", 
                       center=(-R_max, -L_in), 
                       width=W_spiral,
                       orientation=270,
                       layer=(2, 0))
            
        elif direction == 'R':
            c.add_port(name="o1", 
                       center=(R_max, -L_in), 
                       width=W_spiral,
                       orientation=270,
                       layer=(2, 0))
        
        if show_ports == True:
            c.draw_ports()
        
        
        return c

    
    # Used chip size constants.
    W = 10000/4
    H = 5000
    
    c = gf.Component()
    
    # Defining the chip size and an area for it for reference
    #  (could have also added a rectangle)
    #testsp = c << spiral_vortex_beam_emitter_equal_arc_spacing(notch_type="C_out")
    
    #base = c << gf.components.rectangle(size=(W, H), layer=(0, 0))
    
    # Dummy inputs.
    for i in range(1, 40):
        in_i = c << input_coupler()
        in_i.movey(H - i*125)
    
        if i in range(1, 11):
            N = [331, 331, 332, 332, 333, 333, 334, 334, 335, 335]

            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(q=N[i-1], 
                                                                     notch_type="S")
            sp_i.rotate(270)
            sp_i.move((1000+50-0.4, H-i*125+50))
    
        if i in range(11, 21):
            N = [331, 331, 332, 332, 333, 333, 334, 334, 335, 335]

            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(q=N[i-10-1],
                                                                     notch_type="C_in")
            sp_i.rotate(270)
            sp_i.move((1000+50-0.4, H-i*125+50))
            
        if i in range(21, 31):
            N = [331, 331, 332, 332, 333, 333, 334, 334, 335, 335]

            sp_i = c << spiral_vortex_beam_emitter_equal_arc_spacing(q=N[i-20-1], 
                                                                     notch_type="C_out")
            sp_i.rotate(270)
            sp_i.move((1000+50-0.4, H-i*125+50))
    
        if i in range(31, 40):
            N   = [336, 338, 333, 340, 336, 338, 340, 336, 338, 340]
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
    
    c.plot()
    c.show()

    return c


die = nanophotonics_chip()
die.plot()
die.show()
