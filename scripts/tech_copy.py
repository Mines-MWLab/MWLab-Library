## Adding cross sections -ORC

@xsection
def xs_rwg2000(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 2.0,
) -> CrossSection:
    sections = (
        gf.Section(
            width=18,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
    )

@xsection
def xs_rwg750(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 0.72,
) -> CrossSection:
    sections = (
        gf.Section(
            width=18,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
    )

@xsection
def xs_rwg5000(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 5,
) -> CrossSection:
    sections = (
        gf.Section(
            width=18,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
    )

@xsection
def xs_rwg1380(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 1.38,
) -> CrossSection:
    sections = (
        gf.Section(
            width=18,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
    )

@xsection
def xs_rwg250(
    layer: LayerSpec = "LN_RIDGE",
    width: float = 0.25,
) -> CrossSection:
    sections = (
        gf.Section(
            width=18,
            layer="LN_SLAB",
            name="slab",
            simplify=30 * nm,
        ),
    )
    return gf.cross_section.strip(
        width=width,
        layer=layer,
        sections=sections,
    )

@xsection
def xs_swg4000(
    layer: LayerSpec = "LN_SLAB",
    width: float = 4.0,
) -> CrossSection:
    return gf.cross_section.strip(
        width=width,
        layer=layer,
    )

## Done- ORC