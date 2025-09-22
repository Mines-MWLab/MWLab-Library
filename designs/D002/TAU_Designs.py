from functools import partial
from pathlib import Path
import numpy as np
import lnoi400
import gdsfactory as gf
import sys, os
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import devices

gf.clear_cache()

def to_itype(points, dbu):
    return [(int(round(x / dbu)), int(round(y / dbu))) for x, y in points]
dbu = 0.001

@gf.cell
def chip_frame():
    c = gf.get_component("chip_frame", size=(10_000, 5000), center=(5050, 2525))
    return c

chip_layout = chip_frame()

input_ext = 10.0
double_taper = gf.get_component("double_linear_inverse_taper", input_ext=input_ext)

routing_roc = 50.0
ports_gap = 50
frame = 50

# create your new straight OPA waveguide
OPA_straight_waveguide = devices.OPA_straight_waveguide(length=400.0)

# position the OPA waveguide in the layout
OPA_ref = gf.Component()
opa_ref = OPA_ref << OPA_straight_waveguide

# move it horizontally from left edge
left_margin = 50  # distance from left vertical facet
opa_ref.dmovex(left_margin)

# center vertically in the chip
opa_ref.dmovey(chip_layout.dymax / 2)

# Create input/output tapers
# Input taper on the left edge
ec_in = gf.Component()
ec_ref = ec_in << double_taper
ec_ref.dmove(ec_ref.ports["o1"].dcenter, [opa_ref.xmin - input_ext, opa_ref.ports["o1"].center[1]])
ec_in.add_ports(ec_ref.ports)

# Output taper on the right edge
ec_out = gf.Component()
ec_ref = ec_out << double_taper
ec_ref.drotate(180)
ec_ref.dmove(ec_ref.ports["o1"].dcenter, [opa_ref.xmax + input_ext, opa_ref.ports["o2"].center[1]])
ec_out.add_ports(ec_ref.ports)

# assemble final layout
@gf.cell
def die_assembled():
    c = gf.Component()
    c << chip_layout
    c << OPA_ref
    c << ec_in
    c << ec_out

    routing_bend = partial(gf.components.bend_euler, radius=routing_roc, with_arc_floorplan=True)

    gf.routing.route_single(
        c,
        ec_in.ports["o2"],
        opa_ref.ports["o1"],
        cross_section="xs_rwg1000",
        bend=routing_bend,
        straight="straight_rwg1000",
    )

    gf.routing.route_single(
        c,
        ec_out.ports["o2"],
        opa_ref.ports["o2"],
        cross_section="xs_rwg1000",
        bend=routing_bend,
        straight="straight_rwg1000",
    )

    return c

die = die_assembled()
die.plot()
die.show()