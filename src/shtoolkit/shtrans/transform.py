from typing import Literal

import numpy as np

from ._cilm2grid import cilm2grid_fft, cilm2grid_integral
from ._grid2cilm import grid2cilm_fft, grid2cilm_integral

"""
Python wrappers for the spherical harmonic analysis and synthesis
"""

Cilm2GridFunc = {"fft": cilm2grid_fft, "integral": cilm2grid_integral}
Grid2CilmFunc = {"fft": grid2cilm_fft, "integral": grid2cilm_integral}


def cilm2grid(
    cilm: np.ndarray,
    resol: int,
    lmax_calc: int = -1,
    mode: Literal["fft", "integral"] = "fft",
) -> np.ndarray:
    grid = Cilm2GridFunc[mode](cilm, resol, lmax_calc)
    return grid


def grid2cilm(
    grid: np.ndarray,
    lmax_calc: int = -1,
    mode: Literal["fft", "integral"] = "fft",
) -> np.ndarray:
    cilm = Grid2CilmFunc[mode](grid, lmax_calc)
    return cilm
