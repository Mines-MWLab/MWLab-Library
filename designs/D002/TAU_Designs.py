from functools import partial
from pathlib import Path
import numpy as np
import lnoi400
import gdsfactory as gf
import sys, os

# add repo root to path
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import devices

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
input_ext = 10.0
double_taper = gf.get_component("double_linear_inverse_taper", input_ext=input_ext)
routing_roc = 50.0
frame = 50

# minimum required vertical spacing (includes distance to top/bottom facets)
MIN_SPACING = 490.0  # um

# create straight OPA waveguide (single waveguide component)
OPA_straight_waveguide = devices.OPA_straight_waveguide(length=400.0)

@gf.cell
def die_assembled(n_wg: int | None = None, pitch: float = MIN_SPACING) -> gf.Component:
    """
    Place a stack of horizontal straight waveguides centered horizontally on the chip,
    separated vertically by pitch. The pitch must be >= MIN_SPACING.
    If n_wg is None the function will place the maximum number that fits,
    respecting MIN_SPACING between waveguides and between outer waveguides and chip edges.
    """
    if pitch < MIN_SPACING:
        raise ValueError(f"pitch must be >= MIN_SPACING ({MIN_SPACING} µm). Got {pitch}.")

    c = gf.Component()
    c << chip_layout

    # chip usable vertical span H (from bottom facet y=0 to top facet y=dymax)
    H = chip_layout.dymax

    # Formula: must have outer margins >= pitch, and spacing between centers = pitch.
    # Total required height for n waveguides = (n - 1) * pitch + 2 * pitch = (n + 1) * pitch
    # So (n + 1) * pitch <= H  => n <= H / pitch - 1
    max_n = int(np.floor(H / pitch - 1))
    if max_n < 1:
        raise RuntimeError(f"Chip height {H} µm is too small for required pitch {pitch} µm.")

    if n_wg is None:
        n = max_n
    else:
        n = int(n_wg)
        if n > max_n:
            print(f"Requested n_wg={n_wg} exceeds maximum {max_n} for pitch {pitch} -> capping to {max_n}.")
            n = max_n

    # compute Y positions for centers: start at y = pitch, last at y = H - pitch, step = pitch
    y_positions = [pitch + i * pitch for i in range(n)]

    # center horizontally: compute desired x for left edge placement
    x_center = chip_layout.dxmax / 2.0

    routing_bend = partial(
        gf.components.bend_euler,
        radius=routing_roc,
        with_arc_floorplan=True,
    )

    for i in range(n):
        # reference to straight waveguide
        opa_ref = c << OPA_straight_waveguide

        # vertical offset relative to chip center
        dy = (i - (n - 1) / 2) * pitch
        opa_ref.dmovex(chip_layout.dxmax / 2 - opa_ref.xsize / 2)
        opa_ref.dmovey(chip_layout.dymax / 2 + dy)

        # input taper on the left facet
        ec_in = c << double_taper
        ec_in.dmove(
            ec_in.ports["o1"].dcenter,
            [opa_ref.xmin - input_ext - chip_layout.dxmax / 2, opa_ref.ports["o1"].center[1]],
        )

        # output taper on the right facet
        ec_out = c << double_taper
        ec_out.drotate(180)
        ec_out.dmove(
            ec_out.ports["o1"].dcenter,
            [opa_ref.xmax + input_ext + chip_layout.dxmax / 2, opa_ref.ports["o2"].center[1]],
        )

        # routing (straight + bends)
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

    print(f"Placed {n} waveguides with pitch {pitch} µm (chip H={H} µm). Max possible: {max_n}")

    return c

# build and show die (auto-picks max number that fits)
die = die_assembled(n_wg=None, pitch=MIN_SPACING)
die.plot()
die.show()
# _ = die.write_gds(gdsdir=Path.cwd())