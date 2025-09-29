"""Scratch pad for edge coupler development."""

import math
from functools import partial

import gdsfactory as gf

from lnoi400.tech import xs_swg250


def xs_ridge(width: float, layer=(2, 0)) -> gf.cross_section.CrossSection:
    return gf.cross_section.cross_section(width=width, layer=layer)


@gf.cell
def slab_taper_segment(
    tip_length: float = 10.0,
    taper_length: float = 150.0,
    width_start: float = 0.4,
    width_end: float = 6.0,
) -> gf.Component:
    """Slab-layer taper: 10 µm straight tip then sine taper 0.4 → 6 µm."""

    if width_start <= 0 or width_end <= 0:
        raise ValueError("Slab widths must be positive")
    if tip_length < 0 or taper_length <= 0:
        raise ValueError("Lengths must be non-negative (tip) and positive (taper)")

    xs_start = partial(xs_swg250, width=width_start)
    xs_end = partial(xs_swg250, width=width_end)

    slab_tip = gf.components.straight(length=tip_length, cross_section=xs_start)
    slab_taper = gf.components.taper_cross_section(
        cross_section1=xs_start,
        cross_section2=xs_end,
        length=taper_length,
        linear=False,
        width_type="sine",
    )

    c = gf.Component("slab_taper_segment")
    tip_ref = c << slab_tip
    taper_ref = c << slab_taper
    taper_ref.connect("o1", tip_ref.ports["o2"])

    c.add_port("o1", port=tip_ref.ports["o1"])
    c.add_port("o2", port=taper_ref.ports["o2"])
    return c


def _sine_fraction(target_width: float, start: float, end: float) -> float:
    """Returns parametric position (0-1) where sine easing reaches target width."""

    if not (start <= target_width <= end):
        raise ValueError("target width outside taper range")
    ratio = (target_width - start) / (end - start)
    # width = start + (end-start) * (0.5 - 0.5 cos(pi t))
    value = 1 - 2 * ratio
    value = max(-1.0, min(1.0, value))
    return math.acos(value) / math.pi


@gf.cell
def ridge_taper_segment(
    length: float = 100.0,
    width_start: float = 0.25,
    width_end: float = 1.0,
) -> gf.Component:
    if width_start <= 0 or width_end <= 0 or length <= 0:
        raise ValueError("Ridge taper parameters must be positive")

    xs_start = partial(xs_ridge, width=width_start)
    xs_end = partial(xs_ridge, width=width_end)

    taper = gf.components.taper_cross_section(
        cross_section1=xs_start,
        cross_section2=xs_end,
        length=length,
        linear=False,
        width_type="sine",
    )

    c = gf.Component("ridge_taper_segment")
    taper_ref = c << taper
    c.add_ports(taper_ref.ports)
    return c


@gf.cell
def double_taper_edge_coupler(
    slab_tip_length: float = 10.0,
    slab_taper_length: float = 150.0,
    slab_width_start: float = 0.4,
    slab_width_target: float = 0.6,
    slab_width_end: float = 6.0,
    ridge_length: float = 100.0,
    ridge_width_start: float = 0.25,
    ridge_width_end: float = 1.0,
) -> gf.Component:
    slab = slab_taper_segment(
        tip_length=slab_tip_length,
        taper_length=slab_taper_length,
        width_start=slab_width_start,
        width_end=slab_width_end,
    )
    ridge = ridge_taper_segment(
        length=ridge_length,
        width_start=ridge_width_start,
        width_end=ridge_width_end,
    )

    c = gf.Component("double_taper_edge_coupler")
    slab_ref = c << slab

    # Determine x-position where slab reaches target width
    frac = _sine_fraction(slab_width_target, slab_width_start, slab_width_end)
    ridge_start_x = slab_tip_length + frac * slab_taper_length

    ridge_ref = c << ridge
    ridge_ref.dmove((ridge_start_x, 0))

    slab_end_x = slab_tip_length + slab_taper_length
    ridge_end_x = ridge_start_x + ridge_length
    if ridge_end_x < slab_end_x:
        extension_length = slab_end_x - ridge_end_x
        ridge_extension = gf.components.straight(
            length=extension_length,
            cross_section=partial(xs_ridge, width=ridge_width_end),
        )
        ridge_ext_ref = c << ridge_extension
        ridge_ext_ref.connect("o1", ridge_ref.ports["o2"])
        ridge_end_port = ridge_ext_ref.ports["o2"]
    else:
        ridge_end_port = ridge_ref.ports["o2"]

    # update output port to use extension when present
    c.add_port("o2", port=ridge_end_port)

    slab_extent_x = slab_tip_length + slab_taper_length
    slab_rect = gf.components.rectangle(size=(slab_extent_x, 20.0), layer=(3, 1))
    slab_rect_ref = c << slab_rect
    slab_rect_ref.dmove((0, -10.0))

    c.add_port("o1", port=slab_ref.ports["o1"])
    return c


if __name__ == "__main__":
    taper = double_taper_edge_coupler()
    taper.show()
