from functools import partial
from pathlib import Path
import gdsfactory as gf
import sys, os

# add repo root to path
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import TAUdevices, devices 

gf.clear_cache()

# chip frame
@gf.cell
def chip_frame():
    return gf.get_component("chip_frame", size=(10_000, 5000), center=(5050, 2525))

chip_layout = chip_frame()

routing_roc = 50.0
MIN_SPACING = 490.0  # vertical spacing between devices

@gf.cell
def die_assembled() -> gf.Component:
    """
    Place one PM_MLL_cavity_AQ and one AM_MLL_cavity_AQ on the chip, evenly spaced vertically.
    Edge couplers and routing code are included but commented out.
    """
    c = gf.Component()
    c << chip_layout

    # horizontal center
    x_center = chip_layout.dxmax / 2.0

    routing_bend = partial(gf.components.bend_euler, radius=routing_roc, with_arc_floorplan=True)

    # vertical positions
    y_pm = chip_layout.dymax / 2 + MIN_SPACING / 2
    y_am = chip_layout.dymax / 2 - MIN_SPACING / 2

    # place PM device
    pm_ref = c << TAUdevices.PM_MLL_cavity_AQ()
    pm_ref.dmovex(x_center - pm_ref.xsize / 2)
    pm_ref.dmovey(y_pm - pm_ref.dcenter[1])

    # place AM device
    am_ref = c << TAUdevices.AM_MLL_cavity_AQ()
    am_ref.dmovex(x_center - am_ref.xsize / 2)
    am_ref.dmovey(y_am - am_ref.dcenter[1])

    # -------------------------------
    # Edge couplers and routing (optional, commented)
    # input taper for PM
    # ec_in_pm = c << TAUdevices.linear_inverse_taper_AQ()
    # ec_in_pm.dmove(ec_in_pm.ports["o1"].dcenter, [pm_ref.xmin - input_ext, pm_ref.ports["o1"].center[1]])
    # output taper for PM
    # ec_out_pm = c << TAUdevices.linear_inverse_taper_AQ()
    # ec_out_pm.drotate(180)
    # ec_out_pm.dmove(ec_out_pm.ports["o1"].dcenter, [pm_ref.xmax + input_ext, pm_ref.ports["o1"].center[1]])
    # gf.routing.route_single(c, ec_in_pm.ports["o2"], pm_ref.ports["o1"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")
    # gf.routing.route_single(c, ec_out_pm.ports["o2"], pm_ref.ports["o1"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")
    
    # input/output for AM
    # ec_in_am = c << TAUdevices.linear_inverse_taper_AQ()
    # ec_in_am.dmove(ec_in_am.ports["o1"].dcenter, [am_ref.xmin - input_ext, am_ref.ports["o1"].center[1]])
    # ec_out_am = c << TAUdevices.linear_inverse_taper_AQ()
    # ec_out_am.drotate(180)
    # ec_out_am.dmove(ec_out_am.ports["o1"].dcenter, [am_ref.xmax + input_ext, am_ref.ports["o1"].center[1]])
    # gf.routing.route_single(c, ec_in_am.ports["o2"], am_ref.ports["o1"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")
    # gf.routing.route_single(c, ec_out_am.ports["o2"], am_ref.ports["o1"], cross_section="xs_rwg1000", bend=routing_bend, straight="straight_rwg1000")
    # -------------------------------

    return c


# build and show die
die = die_assembled()
die.plot()
die.show()
# _ = die.write_gds(gdsdir=Path.cwd())