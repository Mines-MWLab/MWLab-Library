# TAU_Designs_c2.py
from functools import partial
from pathlib import Path
import numpy as np
import lnoi400
import gdsfactory as gf
import sys, os

# add repo root to path
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Import the laser cell from TAUdevices.py
from scripts.TAUdevices import tunable_mzm_laser_Redwan

gf.clear_cache()

# utility
def to_itype(points, dbu):
    return [(int(round(x / dbu)), int(round(y / dbu))) for x, y in points]

dbu = 0.001

# chip frame
@gf.cell
def chip_frame():
    return gf.get_component("chip_frame", size=(10_000, 5000), center=(5050, 2525))

chip_layout = chip_frame()

# global parameters
input_ext = 10
double_taper = gf.get_component("double_linear_inverse_taper", input_ext=input_ext)
routing_roc = 50.0
frame = 50

# minimum required vertical spacing (kept for parity / future use)
MIN_SPACING = 490.0  # um


@gf.cell
def die_assembled_c2() -> gf.Component:
    """
    Build a die with a single tunable_mzm_laser_Redwan centered on the chip.
    Left/right (west/east) ports are routed to edge couplers on the left and right facets.
    """
    c = gf.Component()
    c << chip_layout

    # Create the laser and place it centered on the chip
    laser = tunable_mzm_laser_Redwan()
    laser_ref = c << laser

    # Center the macro
    laser_ref.dmovex(chip_layout.dxmax / 2 - laser_ref.xsize / 2)
    laser_ref.dmovey(chip_layout.dymax / 2 - laser_ref.ysize / 2)

    # Common routing bend factory
    routing_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
    )

    # Helper to read an angle in degrees for a port (gdsfactory uses 0=east, 90=north, 180=west, 270=south)
    def port_angle_deg(p) -> float:
        # Ports may expose .orientation or .angle depending on version
        return float(getattr(p, "orientation", getattr(p, "angle", 0.0)))

    # Attach edge couplers to any WEST-facing ports (inputs) and EAST-facing ports (outputs)
    # This is generic even if the cell adds/removes ports later.
    for p in laser_ref.ports.values():
        ang = port_angle_deg(p) % 360

        # WEST-facing: place EC on left facet, route to device port
        if np.isclose(ang, 180.0):
            ec_in = c << double_taper
            # position the EC's o1 slightly outside the left chip edge, aligned in Y with the device port
            ec_in.dmove(
                ec_in.ports["o1"].dcenter,
                [-input_ext, p.center[1]],
            )
            # route EC o2 -> device port
            gf.routing.route_single(
                c,
                ec_in.ports["o2"],
                p,
                cross_section="xs_rwg1000",
                bend=routing_bend,
                straight="straight_rwg1000",
            )

        # EAST-facing: place EC on right facet, route to device port
        elif np.isclose(ang, 0.0):
            ec_out = c << double_taper
            ec_out.drotate(180)
            ec_out.dmove(
                ec_out.ports["o1"].dcenter,
                [input_ext + chip_layout.dxmax, p.center[1]],
            )
            gf.routing.route_single(
                c,
                ec_out.ports["o2"],
                p,
                cross_section="xs_rwg1000",
                bend=routing_bend,
                straight="straight_rwg1000",
            )

        # For NORTH/SOUTH ports, we leave them as exposed die ports (could be routed later if needed)

    # Bubble up all ports from the laser instance and any added ECs
    c.add_ports(laser_ref.ports)
    return c


# build and show die
die = die_assembled_c2()
die.plot()
die.show()
# _ = die.write_gds(gdsdir=Path.cwd())