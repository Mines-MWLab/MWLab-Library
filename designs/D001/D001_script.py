from functools import partial
from pathlib import Path
import numpy as np
import lnoi400
import gdsfactory as gf
from scripts import devices

devices.tWave_EOM()

""" @gf.cell
def chip_frame():
    c = gf.get_component("chip_frame", size=(10_000, 5000), center=(0, 0))
    return c

chip_layout = chip_frame()
chip_layout """