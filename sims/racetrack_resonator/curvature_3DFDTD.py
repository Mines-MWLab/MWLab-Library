import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
from functools import partial
from pathlib import Path
import lnoi400

import gplugins as gp
import gplugins.tidy3d as gt
from gplugins import plot
from gplugins.common.config import PATH


component = lnoi400.cells.U_bend_racetrack()
component.show()