from typing import Literal

import numpy as np
import pyfftw

from ._cilm2grid import cilm2grid_engine_by_pocketfft, cilm2grid_engine_by_pyfftw, cilm2grid_integral
from ._grid2cilm import grid2cilm_engine_by_pocketfft, grid2cilm_engine_by_pyfftw, grid2cilm_integral

"""
Python wrappers for the spherical harmonic analysis and synthesis
"""


def cilm2grid(
    cilm: np.ndarray,
    resol: int,
    lmax_calc: int = -1,
    mode: Literal["pyfftw", "pocketfft", "integral"] = "pyfftw",
) -> np.ndarray:
    if mode == "pyfftw":
        nlat = 2 * (resol + 1)
        nlon = 2 * nlat
        input_shape = (nlat, nlon // 2 + 1)
        output_shape = (nlat, nlon)
        if (
            any(item not in globals().keys() for item in ["fftw_c2r_input_shape", "fftw_c2r_output_shape", "fftw_c2r"])
            or globals().get("fftw_c2r_input_shape", 0) != input_shape
            or globals().get("fftw_c2r_output_shape", 0) != output_shape
        ):
            global fftw_c2r_input_shape, fftw_c2r_output_shape, fftw_c2r
            fftw_c2r_input_shape = input_shape
            fftw_c2r_output_shape = output_shape
            fftw_c2r = _alloc_fftw_c2r(fftw_c2r_input_shape, fftw_c2r_output_shape)
        grid = cilm2grid_engine_by_pyfftw(cilm, resol, lmax_calc, fftw_c2r)
    elif mode == "pocketfft":
        grid = cilm2grid_engine_by_pocketfft(cilm, resol, lmax_calc)
    elif mode == "integral":
        grid = cilm2grid_integral(cilm, resol, lmax_calc)
    return grid


def grid2cilm(
    grid: np.ndarray,
    lmax_calc: int = -1,
    mode: Literal["pyfftw", "pocketfft", "integral"] = "pyfftw",
) -> np.ndarray:
    if mode == "pyfftw":
        input_shape = grid.shape
        output_shape = (grid.shape[0], grid.shape[1] // 2 + 1)
        if (
            any(item not in globals().keys() for item in ["fftw_r2c_input_shape", "fftw_r2c_output_shape", "fftw_r2c"])
            or globals().get("fftw_r2c_input_shape", 0) != input_shape
            or globals().get("fftw_r2c_output_shape", 0) != output_shape
        ):
            global fftw_r2c_input_shape, fftw_r2c_output_shape, fftw_r2c
            fftw_r2c_input_shape = input_shape
            fftw_r2c_output_shape = output_shape
            fftw_r2c = _alloc_fftw_r2c(fftw_r2c_input_shape, fftw_r2c_output_shape)
        cilm = grid2cilm_engine_by_pyfftw(grid, lmax_calc, fftw_r2c)
    elif mode == "pocketfft":
        cilm = grid2cilm_engine_by_pocketfft(grid, lmax_calc)
    elif mode == "integral":
        cilm = grid2cilm_integral(grid, lmax_calc)
    return cilm


def _alloc_fftw_r2c(input_shape, output_shape):
    input_array = pyfftw.empty_aligned(input_shape, dtype="float64")
    output_array = pyfftw.empty_aligned(output_shape, dtype="complex128")
    fftw_plan = pyfftw.FFTW(input_array, output_array)
    return fftw_plan


def _alloc_fftw_c2r(input_shape, output_shape):
    input_array = pyfftw.empty_aligned(input_shape, dtype="complex128")
    output_array = pyfftw.empty_aligned(output_shape, dtype="float64")
    fftw_plan = pyfftw.FFTW(input_array, output_array, direction="FFTW_BACKWARD")
    return fftw_plan
