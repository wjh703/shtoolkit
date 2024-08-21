import numpy as np
from pyshtools.expand import MakeGridDH, SHExpandDH


__all__ = ["cilm2grid", "grid2cilm"]


def cilm2grid(
    cilm: np.ndarray,
    resol: int | None = None,
    lmax_calc: int | None = None,
    sampling: int = 2,
) -> np.ndarray:
    grid = MakeGridDH(cilm, sampling=sampling, lmax=resol, lmax_calc=lmax_calc, extend=False)
    return grid


def grid2cilm(grd: np.ndarray, lmax_calc: int | None = None, sampling: int = 2) -> np.ndarray:
    cilm = SHExpandDH(grd, sampling=sampling, lmax_calc=lmax_calc)
    return cilm
