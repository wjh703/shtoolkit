from typing import Literal

import numpy as np

from ._cilm2grid import cilm2grid_fft, cilm2grid_integral
from ._grid2cilm import grid2cilm_fft, grid2cilm_integral

from .legendre import fnALFs_cache

"""
Python wrappers for the spherical harmonic analysis and synthesis
"""

__all__ = ["cilm2grid", "grid2cilm"]


Cilm2GridFunc = {"fft": cilm2grid_fft, "integral": cilm2grid_integral}
Grid2CilmFunc = {"fft": grid2cilm_fft, "integral": grid2cilm_integral}


def cilm2grid(
    cilm: np.ndarray,
    resol: int,
    lmax_calc: int = -1,
    pilm: np.ndarray | None = None,
    mode: Literal["fft", "integral"] = "fft",
) -> np.ndarray:
    if pilm is None:
        lmax = lmax_calc if lmax_calc >= 0 else cilm.shape[1] - 1
        rad_colat = tuple(np.linspace(0, np.pi, 2 * (resol + 1), endpoint=False))
        pilm = fnALFs_cache(rad_colat, lmax)
    grid = Cilm2GridFunc[mode](cilm, resol, lmax_calc, pilm)
    return grid


def grid2cilm(
    grid: np.ndarray,
    lmax_calc: int = -1,
    pilm: np.ndarray | None = None,
    mode: Literal["fft", "integral"] = "fft",
) -> np.ndarray:
    if pilm is None:
        lmax = lmax_calc if lmax_calc >= 0 else grid.shape[0] // 2 - 1
        rad_colat = tuple(np.linspace(0, np.pi, grid.shape[0], endpoint=False))
        pilm = fnALFs_cache(rad_colat, lmax).transpose(1, 2, 0)
    cilm = Grid2CilmFunc[mode](grid, lmax_calc, pilm)
    return cilm
