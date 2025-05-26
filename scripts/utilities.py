import gdsfactory as gf
import lnoi400
from gplugins.common.config import PATH
from gdsfactory.typings import CrossSectionSpec
from lnoi400.spline import bend_S_spline_varying_width
import numpy as np

from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path