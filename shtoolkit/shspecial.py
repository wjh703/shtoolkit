import re

import numpy as np
import tqdm
from pyshtools.shio import SHctor, SHrtoc

from .shtrans import cilm2grid, grid2cilm, fnalf
from .shunit import convert, SH_CONST, mass2geo, mass2upl
from .shtype import SpharmUnit, LoadLoveNumDict, LeakCorrMethod, MassConserveMode
from .shfilter import gauss_smooth, fan_smooth

__all__ = ["uniform_distributed", "sea_level_equation"]


def uniform_distributed(
    load_data: np.ndarray,
    oceanmask: np.ndarray,
) -> np.ndarray:
    load_cilm = grid2cilm(load_data, 0)
    oc_cilm = grid2cilm(oceanmask, 0)
    load_conserve = -load_cilm[0, 0, 0] / oc_cilm[0, 0, 0] * oceanmask + load_data
    return load_conserve


def sea_level_equation(
    load_data: np.ndarray,
    oceanmask: np.ndarray,
    lln: LoadLoveNumDict,
    lmax: int,
    unit: SpharmUnit,
    rot: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    k = 0
    k_max = 10
    epsilon = 1e-5
    chi = 3 * epsilon

    resol = load_data.shape[0] // 2 - 1

    load_cilm = grid2cilm(load_data, lmax)
    load_cilm = convert(load_cilm, unit, "mewh", lln)

    oc_cilm = grid2cilm(oceanmask, lmax)

    S_cilm = -load_cilm[0, 0, 0] / oc_cilm[0, 0, 0] * oc_cilm

    ewh2mass_coef = 1000
    mass2geo_coef = mass2geo(lmax, lln)
    mass2upl_coef = mass2upl(lmax, lln)

    while chi >= epsilon and k < k_max:
        T_cilm_h = load_cilm + S_cilm
        T_cilm = T_cilm_h * ewh2mass_coef
        T_geo_cilm = T_cilm * mass2geo_coef
        T_upl_cilm = T_cilm * mass2upl_coef
        # T_geo_cilm = convert(T_cilm, 'kgm2mass', 'mgeo', lln)
        # T_upl_cilm = convert(T_cilm, 'kgm2mass', 'mupl', lln)

        if rot:
            rot_geo_cilm, rot_upl_cilm = _calc_rot(T_cilm)
            T_geo_cilm += rot_geo_cilm
            T_upl_cilm += rot_upl_cilm

        T_rsl_cilm = T_geo_cilm - T_upl_cilm
        T_rsl = cilm2grid(T_rsl_cilm, resol, lmax)

        RO = T_rsl * oceanmask
        RO_cilm = grid2cilm(RO, lmax)

        delPhi_g = -(load_cilm[0, 0, 0] + RO_cilm[0, 0, 0]) / oc_cilm[0, 0, 0]

        S_cilm_new = RO_cilm + delPhi_g * oc_cilm
        chi = np.abs(
            (np.square(S_cilm_new).sum() - np.square(S_cilm).sum()) / np.square(S_cilm).sum()
        )
        S_cilm = S_cilm_new.copy()
        k += 1

    sea_level_fingerprint = cilm2grid(S_cilm, resol, lmax) * oceanmask
    geo_conserve = cilm2grid(T_geo_cilm, resol, lmax)
    upl_conserve = cilm2grid(T_upl_cilm, resol, lmax)
    load_conserve = cilm2grid(convert(S_cilm, "mewh", unit, lln), resol, lmax) + load_data
    return sea_level_fingerprint, geo_conserve, upl_conserve, load_conserve


def _calc_rot(L_cilm_real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = SH_CONST["a"]
    k2e = SH_CONST["k2e"]
    k2e_t = SH_CONST["k2e_t"]
    h2e_t = SH_CONST["h2e_t"]
    omega = SH_CONST["omega"]
    cw_freq = SH_CONST["cw_freq"]
    moi_a = SH_CONST["moi_a"]
    moi_c = SH_CONST["moi_c"]
    gave = SH_CONST["gave"]

    L_cilm_complex = SHrtoc(L_cilm_real)
    J13 = a**4 * np.pi * (32 / 15) ** 0.5 * L_cilm_complex[0, 2, 1]
    J23 = -(a**4) * np.pi * (32 / 15) ** 0.5 * L_cilm_complex[1, 2, 1]
    J33 = -(a**4) * np.pi * (8 / 3 / 5**0.5) * L_cilm_complex[0, 2, 0]

    m1 = omega * (1 + k2e) / (moi_a * cw_freq) * J13
    m2 = omega * (1 + k2e) / (moi_a * cw_freq) * J23
    m3 = -(1 + k2e) / moi_c * J33
    # m3 = 0
    La_cilm_complex = np.zeros_like(L_cilm_complex)
    # La00 = -(1+k2e) / moi_C * a**4 * np.pi * (8/3/5**0.5) * -L_ilm[0, 2, 0] * a**2 * omega**2 * 2 / 3
    # Calculate rotational potential, refer to Kindall et al. (2005).
    # Note that some modifications have been applied to match the pyshtool's CSH format.
    La_cilm_complex[0, 0, 0] = a**2 * omega**2 / 3 * (m1**2 + m2**2 + m3**2 + 2 * m3)
    La_cilm_complex[0, 2, 0] = a**2 * omega**2 / (6 * 5**0.5) * (m1**2 + m2**2 - 2 * m3**2 - 4 * m3)
    La_cilm_complex[0, 2, 1] = a**2 * omega**2 / 30**0.5 * (m1 * (1 + m3))
    La_cilm_complex[1, 2, 1] = a**2 * omega**2 / 30**0.5 * (-m2 * (1 + m3))
    La_cilm_complex[0, 2, 2] = a**2 * omega**2 / (5**0.5 * 24**0.5) * (m2**2 - m1**2)
    La_cilm_complex[1, 2, 2] = a**2 * omega**2 / (5**0.5 * 24**0.5) * (2 * m1 * m2)
    La_cilm_real = SHctor(La_cilm_complex)

    indices = tuple(zip([0, 2, 0], [0, 2, 1], [1, 2, 1], [0, 2, 2], [1, 2, 2]))
    La2geo = np.zeros_like(La_cilm_real)
    La2geo[0, 0, 0] = 1 / gave
    La2geo[*indices] = (1 + k2e_t) / gave
    La_geo = La_cilm_real * La2geo

    La2upl = np.zeros_like(La_cilm_real)
    La2upl[*indices] = h2e_t / gave
    La_upl = La_cilm_real * La2upl

    return La_geo, La_upl


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

    leakcorr_method = re.findall(r"buf|FM", leakage["method"])
    smooth_kind = re.findall(r"gs|fs", leakage["method"])
    smooth_coef_func = {"gs": gauss_smooth, "fs": fan_smooth}
    if smooth_kind:
        radius = leakage["radius"]
        if radius is None:
            msg = "The key value of 'radius' in 'leakage' needs 'int' object, rather than None."
            raise ValueError(msg)
        coeffg = smooth_coef_func[smooth_kind[0]](lmax, radius)
    else:
        coeffg = None

    calc_indices = tuple(zip((0, 1, 0), (0, 1, 1), (1, 1, 1)))
    setzero_indices = tuple(zip((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)))

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

    pilm = fnalf(rad_colat, 1)  # type: ignore
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
            cilm_mas_i = convert(cilm_i * coeffg, unit, "kgm2mass", lln)
        else:
            cilm_mas_i = convert(cilm_i, unit, "kgm2mass", lln)
        mas_i = cilm2grid(cilm_mas_i, resol, lmax)
        ocean_mas_i = mas_i * oceanmask
        land_mas_i = mas_i * landmask

        # if leakcorr_method == 'FM' and coeffg is not None:
        #     land_mas_i = _FM(land_mas_i, land_msk, oc_msk, sph, coeffg, cilm_set_zero=cilm_set_zero)  # type: ignore
        #     clm_leak_i = sph.grd2spec(land_mas_i) * coeffg
        #     leak_mas_i = sph.spec2grd(clm_leak_i) * oc_msk
        #     ocean_mas_i -= leak_mas_i

        g_iv = grid2cilm(ocean_mas_i, 2)[*calc_indices]

        for j in range(4):
            if "sal" in mode:
                # ex_mas_i = sea_level_equation(land_mas_i, oceanmask, lln, lmax, "kgm2mass")[-1] * oceanmask

                ex_ewh_i = sea_level_equation(land_mas_i, oceanmask, lln, lmax, "kgm2mass")[0]
                ex_mas_i = ex_ewh_i * 1000
            else:
                ex_mas_i = uniform_distributed(land_mas_i, oceanmask) * oceanmask
            ex_iv = grid2cilm(ex_mas_i, 2)[*calc_indices]
            x = np.linalg.solve(im, ex_iv - g_iv)

            cilm_mas_i[*calc_indices] = x

            if j + 1 < 4:
                land_mas_i = cilm2grid(cilm_mas_i, resol, lmax) * landmask
                # if leakcorr_method == 'FM':
                #     land_mas_i = _FM(sph.spec2grd(cilm_mas_i)*land_msk, land_msk, oc_msk, sph, coeffg)  # type: ignore
                # else:
                #     land_mas_i = sph.spec2grd(cilm_mas_i)*land_msk
        cilm[i, *calc_indices] = convert(cilm_mas_i, "kgm2mass", unit, lln)[*calc_indices]
    return cilm
