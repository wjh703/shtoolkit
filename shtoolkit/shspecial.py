import numpy as np
from pyshtools.shio import SHctor, SHrtoc

from .shtrans import cilm2grid, grid2cilm
from .shunit import convert, SH_CONST, mass2geo, mass2upl
from .shtype import SpharmUnit, LoadLoveNumDict


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

    sea_level_fingerprint = RO
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
