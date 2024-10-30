import re

import numpy as np
import tqdm

from ..shfilter import fan_smooth, gauss_smooth
from ..shtrans import cilm2grid, fnALFs, grid2cilm
from ..shtype import (
    LeakCorrMethod,
    LoadLoveNumDict,
    MassConserveMode,
    SpharmUnit,
)
from ..shunit import unitconvert
from .leakage import foward_modeling
from .sea_level_equation import non_uniform, uniform


def standard(
    coeffs: np.ndarray,
    unit: SpharmUnit,
    oceanmask: np.ndarray,
    lln: LoadLoveNumDict,
    lmax: int | None = None,
    leakage: LeakCorrMethod = {"method": "buf", "radius": None},
    mode: MassConserveMode = "sal_rot",
):
    if lmax is None:
        lmax = coeffs.shape[-2] - 1
    resol = oceanmask.shape[0] // 2 - 1

    leakcorr_method = re.search(r"buf|FM", leakage["method"]).group()  # type: ignore
    smooth_kind = re.search(r"gs|fs", leakage["method"]).group()  # type: ignore
    smooth_coef_func = {"gs": gauss_smooth, "fs": fan_smooth}
    if smooth_kind:
        radius = leakage["radius"]
        if radius is None:
            msg = "The key value of 'radius' in 'leakage' needs 'int' object, rather than None."
            raise ValueError(msg)
        coeffg = smooth_coef_func[smooth_kind](lmax, radius)
    else:
        coeffg = None

    calc_indices = tuple(zip((0, 1, 0), (0, 1, 1), (1, 1, 1), strict=False))
    setzero_indices = tuple(zip((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1), strict=False))

    cilm = np.copy(coeffs)
    cilm[:, *setzero_indices] = 0

    landmask = 1.0 - oceanmask

    lat = np.linspace(90, -89, oceanmask.shape[0])
    lon = np.linspace(0, 359, oceanmask.shape[1])
    nlat = len(lat)
    nlon = len(lon)
    rad_colat = np.deg2rad(90 - lat)
    rad_lon = np.deg2rad(lon)
    solid_angle = np.sin(rad_colat) * np.deg2rad(180 / nlat) * np.deg2rad(360 / nlon)

    pilm = fnALFs(rad_colat, 1)  # type: ignore
    p10 = np.squeeze(pilm[:, 1, 0])
    p11 = np.squeeze(pilm[:, 1, 1])
    cos0phi = np.cos(0 * rad_lon)
    cos1phi = np.cos(1 * rad_lon)
    sin1phi = np.sin(1 * rad_lon)
    im = np.zeros((3, 3))
    for i in range(nlat):
        pc10 = p10[i] * cos0phi
        pc11 = p11[i] * cos1phi
        ps11 = p11[i] * sin1phi
        im[0, 0] += np.sum(pc10 * oceanmask[i] * pc10 * solid_angle[i])
        im[0, 1] += np.sum(pc10 * oceanmask[i] * pc11 * solid_angle[i])
        im[0, 2] += np.sum(pc10 * oceanmask[i] * ps11 * solid_angle[i])
        im[1, 0] += np.sum(pc11 * oceanmask[i] * pc10 * solid_angle[i])
        im[1, 1] += np.sum(pc11 * oceanmask[i] * pc11 * solid_angle[i])
        im[1, 2] += np.sum(pc11 * oceanmask[i] * ps11 * solid_angle[i])
        im[2, 0] += np.sum(ps11 * oceanmask[i] * pc10 * solid_angle[i])
        im[2, 1] += np.sum(ps11 * oceanmask[i] * pc11 * solid_angle[i])
        im[2, 2] += np.sum(ps11 * oceanmask[i] * ps11 * solid_angle[i])
    im /= 4.0 * np.pi

    for i, cilm_i in enumerate(tqdm.tqdm(cilm, desc="calc_deg1")):
        if coeffg is not None:
            cilm_mas_i = unitconvert(cilm_i * coeffg, unit, "kgm2mass", lln)
        else:
            cilm_mas_i = unitconvert(cilm_i, unit, "kgm2mass", lln)
        mas_i = cilm2grid(cilm_mas_i, resol, lmax)
        ocean_mas_i = mas_i * oceanmask
        land_mas_i = mas_i * landmask

        if leakcorr_method == "FM":
            land_mas_i = foward_modeling(
                land_mas_i,
                landmask,
                oceanmask,
                smooth_kind,  # type: ignore
                radius,  # type: ignore
                lmax,
                setzero_indices,
            )
            cilm_leakage_i = grid2cilm(land_mas_i, lmax) * coeffg
            leak_mas_i = cilm2grid(cilm_leakage_i, resol, lmax) * oceanmask
            ocean_mas_i -= leak_mas_i  # type: ignore

        g_iv = grid2cilm(ocean_mas_i, 2)[*calc_indices]

        for j in range(4):
            if "sal" in mode:
                is_rot = "rot" in mode
                ex_ewh_i = non_uniform(land_mas_i, oceanmask, lln, lmax, "kgm2mass", is_rot)[0]
                ex_mas_i = ex_ewh_i * 1000
            else:
                ex_mas_i = uniform(land_mas_i, oceanmask) * oceanmask
            ex_iv = grid2cilm(ex_mas_i, 2)[*calc_indices]
            x = np.linalg.solve(im, ex_iv - g_iv)

            cilm_mas_i[*calc_indices] = x

            if j + 1 < 4:
                land_mas_i = cilm2grid(cilm_mas_i, resol, lmax) * landmask
                if leakcorr_method == "FM":
                    land_mas_i = foward_modeling(
                        land_mas_i,
                        landmask,
                        oceanmask,
                        smooth_kind,  # type: ignore
                        radius,  # type: ignore
                        lmax,
                    )

        cilm[i, *calc_indices] = unitconvert(cilm_mas_i, "kgm2mass", unit, lln)[*calc_indices]
    return cilm
