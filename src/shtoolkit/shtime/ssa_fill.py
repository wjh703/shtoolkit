from typing import TypedDict, Sequence
from functools import partial

import numpy as np
from scipy.signal import periodogram

__all__ = ["ssa_gap_filling"]


class SHcoeffWithGap(TypedDict):
    coeff: np.ndarray
    do: tuple[int, int]


# def _shc_gap_filling(
#         dtime: np.ndarray,
#         shc: SHcoeffWithGap | Sequence[SHcoeffWithGap],
#     ) -> np.ndarray:

#     if isinstance(shc, dict):
#         x_filling = _shc_gap_filling(dtime, shc)
#     elif isinstance(shc, list):
#         max_workers = 3
#         x_filling = np.array(process_map(partial(_shc_gap_filling, dtime), shc, max_workers=max_workers))
#     else:
#         msg = f'input shc is not {SHCGapSeries} typeddict'
#         raise ValueError(msg)

#     return x_filling


def ssa_gap_filling(dtime: np.ndarray, series: SHcoeffWithGap) -> np.ndarray:
    """
    filling the SHC gap within GRACE/GFO
    """
    x = series["coeff"]
    if x.size != dtime.size:
        raise AttributeError(f"x.size:{x.size} != dtime.size:{dtime.size}")

    # GRACE时期插值
    m1, k1 = 24, 12
    t1 = dtime < 2017.5
    # for i in tqdm(range(1), desc='gap_filling_a', leave=False):
    #
    x[t1] = _ssa_iter(x[t1], m1, k1)

    # GRACE-FO时期插值
    t2 = (dtime > 2003) & (dtime < 2017)
    tt2 = np.floor(dtime[t2])
    x2 = x[t2]
    m2_bound = np.arange(12, 73, 12)
    deg, ord = series["do"]
    if ord <= 40:
        kk = (4 * (1 - (ord / 40) ** 2) ** 2 + 3) * (1 - (deg / 60) ** 2 / 2)
    else:
        kk = 3 * (1 - (deg / 60) ** 2 / 2)
    k2 = round(kk)

    # M2的最佳参数设置
    gap_yrs = np.arange(2004, 2016)
    gap_error = np.zeros(len(m2_bound), dtype=np.float64)

    for i, m in enumerate(m2_bound):
        diff_yr = np.zeros(len(gap_yrs), dtype=np.float64)
        for j, yr in enumerate(gap_yrs):
            igap = tt2 == yr
            x3 = np.copy(x2)
            x3[igap] = np.nan
            x3 = _ssa_iter(x3, m, k2)
            diff_yr[j] = np.sqrt(np.square(x2[igap] - x3[igap]).mean())
        gap_error[i] = np.sqrt(np.mean(diff_yr**2))

    m2 = m2_bound[np.argmin(gap_error)]
    x = _ssa_iter(x, m2, k2)

    return x


def _ssa_iter(x: np.ndarray, M: int, K: int) -> np.ndarray:
    """
    迭代奇异谱分析插值 + 分布积累函数检验
    """
    eplision = 1e-2
    gap = np.isnan(x)
    x[gap] = 1e-20
    for i in range(K):
        chi = 3 * eplision
        while chi > eplision:
            x_rec, x_rc = _ssa(x, M, i + 1)
            chi = np.sqrt(np.square(x[gap] - x_rec[gap]).mean() / np.square(x[gap]).mean())
            x[gap] = x_rec[gap]
    x_rec_cdf, _ = _cdf(x_rc)
    x[gap] = x_rec_cdf[gap]
    return x


def _ssa(x: np.ndarray, M: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    奇异谱分解
    """
    N = x.size
    L = N - M + 1
    Y = np.zeros((M, L), dtype=np.float64)
    for i in range(L):
        Y[:, i] = x[i : i + M]

    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    Z: list[np.ndarray] = [s[i] * U[:, i][:, np.newaxis] @ Vh[i, :][np.newaxis, :] for i in range(K)]

    # r = np.linalg.matrix_rank(Y)

    rc = np.zeros((N, K), dtype=np.float64)
    for i, xs in enumerate(Z):
        rc[:, i] = [np.mean(xs[::-1, :].diagonal(i)) for i in range(-xs.shape[0] + 1, xs.shape[1])]
    x_rec = rc.sum(axis=1).ravel()

    return x_rec, rc


def _cdf(rc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    cumulative distribution function
    """
    freq, pow = periodogram(rc, axis=0, fs=12)
    cdf = np.cumsum(pow, axis=0)
    cdf /= cdf.max(axis=0)
    idx = np.argmin(np.abs(freq - 3))
    rc_new = rc[:, cdf[idx] >= 0.9]
    rec_new = rc_new.sum(axis=1).ravel()

    return rec_new, rc_new
