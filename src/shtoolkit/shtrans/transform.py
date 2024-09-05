from typing import Literal

import numpy as np

from ._cilm2grid import cilm2grid_fft, cilm2grid_integral
from ._grid2cilm import grid2cilm_fft, grid2cilm_integral

"""
Python wrappers for the spherical harmonic analysis and synthesis
"""

__all__ = ["cilmtogrid", "gridtocilm"]


Cilm2GridFunc = {
    'fft': cilm2grid_fft,
    'integral': cilm2grid_integral
}
Grid2CilmFunc = {
    'fft': grid2cilm_fft,
    'integral': grid2cilm_integral
}

def cilmtogrid(
    cilm: np.ndarray,
    resol: int,
    lmax_calc: int = -1,
    pilm: np.ndarray | None = None,
    mode: Literal['fft', 'integral'] = 'fft'
) -> np.ndarray:
    grid = Cilm2GridFunc[mode](cilm, resol, lmax_calc, pilm)
    return grid


def gridtocilm(
    grid: np.ndarray, 
    lmax_calc: int = -1,
    pilm: np.ndarray | None = None,
    mode: Literal['fft', 'integral'] = 'fft'
) -> np.ndarray:
    if pilm is not None:
        cilm = Grid2CilmFunc[mode](grid, lmax_calc, pilm.transpose(1, 2, 0))
    else:
        cilm = Grid2CilmFunc[mode](grid, lmax_calc)
    return cilm


