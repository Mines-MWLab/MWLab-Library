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
    width: float = 0.75,
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


## Done- ORC