from typing import Sequence

import numpy as np

from ..shfilter import fan_smooth, gauss_smooth
from ..shtrans import cilm2grid, grid2cilm
from ..shtype import SHSmoothKind, LoadLoveNumDict
from ..shunit import unitconvert
from .sea_level_equation import uniform, non_uniform


def forward_modeling(
    load_data: np.ndarray,
    loadmask: np.ndarray,
    oceanmask: np.ndarray,
    smooth: SHSmoothKind,
    radius: int,
    lmax: int | None = None,
    setzero_indices: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
) -> np.ndarray:
    if not (np.allclose(load_data.shape, loadmask.shape) and np.allclose(oceanmask.shape, loadmask.shape)):
        msg = "The shape of 'load_data', 'landmask' and 'oceanmask' are unequal"
        raise ValueError(msg)
    resol = load_data.shape[0] // 2 - 1
    if lmax is None:
        lmax = resol
    smooth_coef_func = {"gauss": gauss_smooth, "fan": fan_smooth}
    coeffg = smooth_coef_func[smooth](lmax, radius)
    # 将真值进行一次截断和滤波作为循环初
    obs = load_data
    m_tru = np.copy(obs)

    for i in range(100):
        glo_grid = uniform(m_tru, oceanmask)
        glo_cilm = grid2cilm(glo_grid, lmax)

        glo_cilm_smoothed = glo_cilm * coeffg
        glo_cilm_smoothed[*setzero_indices] = 0.0

        pre = cilm2grid(glo_cilm_smoothed, resol, lmax) * loadmask
        delta = obs - pre
        if (np.abs(delta * loadmask) < 0.5).all():
            print("Converged in %d iterations" % (i + 1))
            break
        m_tru += delta * 1.2
        # rmse = np.sqrt(np.sum(delta ** 2) / delta[delta != 0].size)
        # print(rmse)
    return m_tru


def forward_modeling1(
    load_data: np.ndarray,
    loadmask: np.ndarray,
    oceanmask: np.ndarray,
    smooth: SHSmoothKind,
    radius: int,
    lmax: int | None = None,
    setzero_indices: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
) -> np.ndarray:
    if not (np.allclose(load_data.shape, loadmask.shape) and np.allclose(oceanmask.shape, loadmask.shape)):
        msg = "The shape of 'load_data', 'landmask' and 'oceanmask' are unequal"
        raise ValueError(msg)
    resol = load_data.shape[0] // 2 - 1
    if lmax is None:
        lmax = resol
    smooth_coef_func = {"gauss": gauss_smooth, "fan": fan_smooth}
    coeffg = smooth_coef_func[smooth](lmax, radius)
    # 将真值进行一次截断和滤波作为循环初
    obs = uniform(load_data, oceanmask)
    # obs = load_data
    m_tru = np.copy(obs)

    for i in range(100):
        # glo_grid = uniform(m_tru, oceanmask)
        glo_cilm = grid2cilm(m_tru, lmax)

        glo_cilm_smoothed = glo_cilm * coeffg
        glo_cilm_smoothed[*setzero_indices] = 0.0

        pre = cilm2grid(glo_cilm_smoothed, resol, lmax)
        delta = obs - pre
        if (np.abs(delta * loadmask) < 0.5).all():
            print("Converged in %d iterations" % (i + 1))
        m_tru += delta * 1
        m_tru = uniform(m_tru * loadmask, oceanmask)
        # rmse = np.sqrt(np.sum(delta ** 2) / delta[delta != 0].size)
        # print(rmse)
    return m_tru


def gloabl_forward_modeling(
    cilm: np.ndarray,
    lln: LoadLoveNumDict,
    oceanmask: np.ndarray,
    smooth: SHSmoothKind,
    radius: int,
    resol: int | None = None,
    # lmax: int | None = None,
    setzero_indices: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
    iterations: int = 100,
) -> np.ndarray:
    
    lmax: int = cilm.shape[1] - 1
    if resol is None:
        resol = (lmax + 1) * 2

    landmask = 1 - oceanmask
    smooth_coef_func = {"gauss": gauss_smooth, "fan": fan_smooth}
    coeffg = smooth_coef_func[smooth](lmax, radius)

    cilm_mass = unitconvert(cilm, 'stokes', 'mmewh', lln)
    cilm_mass_smoothed = cilm_mass * coeffg
    cilm_mass_smoothed[setzero_indices] = 0.0
    grace_apparent_mass = cilm2grid(cilm_mass_smoothed, resol, lmax)
    # breakpoint()
    # grace_true_mass = uniform(grace_apparent_mass * landmask, oceanmask)
    _, _, _, grace_true_mass = non_uniform(grace_apparent_mass * landmask, oceanmask, lln, lmax, 'mmewh')
    for i in range(iterations):
        cilm_mass_foward = grid2cilm(grace_true_mass, lmax)
        cilm_mass_foward_smoothed = cilm_mass_foward * coeffg
        cilm_mass_foward_smoothed[setzero_indices] = 0.0
        grace_apparent_mass_foward = cilm2grid(cilm_mass_foward_smoothed, resol, lmax)

        diff = grace_apparent_mass - grace_apparent_mass_foward
        if (np.abs(diff * landmask) < 5).all():
            print("Converged in %d iterations" % (i + 1))
            break
        grace_true_mass += diff * 1.2

        # grace_true_mass = uniform(grace_true_mass * landmask, oceanmask)
        _, _, _, grace_true_mass = non_uniform(grace_true_mass * landmask, oceanmask, lln, lmax, 'mmewh')
    # return cilm2grid(cilm_mass_foward, resol, lmax)
    print(i)
    return grace_true_mass * landmask



   
# def forward_modeling_improved(
#     load_data: np.ndarray,
#     loadmask: np.ndarray,
#     oceanmask: np.ndarray,
#     smooth: SHSmoothKind,
#     radius: int,
#     lmax: int | None = None,
#     setzero_indices: Sequence[int] | Sequence[Sequence[int]] = (0, 0, 0),
# ) -> np.ndarray:
#     if not (np.allclose(load_data.shape, loadmask.shape) and np.allclose(oceanmask.shape, loadmask.shape)):
#         msg = "The shape of 'load_data', 'landmask' and 'oceanmask' are unequal"
#         raise ValueError(msg)
#     resol = load_data.shape[0] // 2 - 1
#     if lmax is None:
#         lmax = resol
#     smooth_coef_func = {"gauss": gauss_smooth, "fan": fan_smooth}
#     coeffg = smooth_coef_func[smooth](lmax, radius)
#     # 将真值进行一次截断和滤波作为循环初
#     obs = load_data
#     m_tru = np.copy(obs)

#     for _ in range(100):
#         glo_grid = uniform(m_tru, oceanmask)
#         glo_grid = non_uniform(m_tru, oceanmask, lln, lmax, )
#         glo_cilm = grid2cilm(glo_grid, lmax)

#         glo_cilm_smoothed = glo_cilm * coeffg
#         glo_cilm_smoothed[*setzero_indices] = 0.0

#         pre = cilm2grid(glo_cilm_smoothed, resol, lmax) * loadmask
#         delta = obs - pre
#         m_tru += delta * 1.2
#         # rmse = np.sqrt(np.sum(delta ** 2) / delta[delta != 0].size)
#         # print(rmse)
#     return m_tru