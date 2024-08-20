import numpy as _np
from pyshtools.expand import (      
        MakeGridDH as _MakeGridDH,  
        SHExpandDH as _SHExpandDH 
    )


__all__ = ['cilm2grid', 'grid2cilm']


def cilm2grid(
        cilm: _np.ndarray,
        resol: int | None = None,
        lmax_calc: int | None = None,
        sampling: int = 2
    ) -> _np.ndarray:
    grid = _MakeGridDH(cilm, sampling=sampling, lmax=resol, lmax_calc=lmax_calc, extend=False)
    return grid


def grid2cilm(
        grd: _np.ndarray,
        lmax_calc: int | None = None,
        sampling: int = 2
    ) -> _np.ndarray:
    cilm = _SHExpandDH(grd, sampling=sampling, lmax_calc=lmax_calc)
    return cilm