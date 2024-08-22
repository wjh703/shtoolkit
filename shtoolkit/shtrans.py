from typing import Sequence

import numpy as np
from pyshtools.expand import MakeGridDH, SHExpandDH
from pyshtools.legendre import legendre


__all__ = ["cilm2grid", "grid2cilm", "fnalf"]


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


def fnalf(rad_colat: np.ndarray | list | float, lmax: int) -> np.ndarray:
    if isinstance(rad_colat, float):
        return legendre(lmax, np.cos(rad_colat))
    elif isinstance(rad_colat, np.ndarray | list):
        pilm = []
        for theta in rad_colat:
            pilm.append(legendre(lmax, np.cos(theta)))
        return np.asarray(pilm)
    else:
        msg = "'rad_colat' type should be 'Sequence[float]' or 'float' object"
        raise ValueError(msg)
