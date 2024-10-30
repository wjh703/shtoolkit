from typing import Sequence

import numpy as np

from ..shfilter import fan_smooth, gauss_smooth
from ..shtrans import cilm2grid, grid2cilm
from ..shtype import SHSmoothKind
from .sea_level_equation import uniform


def foward_modeling(
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
    smooth_coef_func = {"gs": gauss_smooth, "fs": fan_smooth}
    coeffg = smooth_coef_func[smooth](lmax, radius)
    # 将真值进行一次截断和滤波作为循环初
    obs = load_data
    m_tru = np.copy(obs)

    for _ in range(100):
        glo_grid = uniform(m_tru, oceanmask)
        glo_cilm = grid2cilm(glo_grid, lmax)

        glo_cilm_smoothed = glo_cilm * coeffg
        glo_cilm_smoothed[*setzero_indices] = 0.0

        pre = cilm2grid(glo_cilm_smoothed, resol, lmax) * loadmask
        delta = obs - pre
        m_tru += delta * 1.2
        # rmse = np.sqrt(np.sum(delta ** 2) / delta[delta != 0].size)
        # print(rmse)
    return m_tru
