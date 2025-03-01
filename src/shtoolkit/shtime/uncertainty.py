import numpy as np
from scipy.optimize import minimize
from tqdm.contrib.concurrent import process_map


def gtch_map(x: np.ndarray, max_workers: int = 3, chunksize: int = 100) -> np.ndarray:
    nrow, ncol = x.shape[-2], x.shape[-1]
    x = x.reshape((x.shape[0], x.shape[1], -1))
    egrd = np.asarray(process_map(gtch, x.T, max_workers=max_workers, chunksize=chunksize))

    return egrd.T.reshape((-1, nrow, ncol))


def btch_map(x: np.ndarray, max_workers: int = 3, chunksize: int = 4) -> tuple[np.ndarray, np.ndarray]:
    sample, nrow, ncol = x.shape[1], x.shape[2], x.shape[3]
    x = x.reshape((x.shape[0], x.shape[1], -1))
    result_lst = process_map(btch, x.T[9000:], max_workers=max_workers, chunksize=chunksize)
    comb_grd, egrd = map(np.array, zip(*result_lst, strict=False))
    egrd = egrd.T.reshape((-1, nrow, ncol))
    comb_grd = comb_grd.reshape((sample, nrow, ncol))

    return comb_grd, egrd


def gtch(x: np.ndarray) -> np.ndarray:
    """Generalized Three-Cornered Hat (GTCH) Method
    It can quantify the uncertainties of time series when the truth value is unknown
    There need at least 3 sets of time series to run this method"""
    # if x.shape[0] < x.shape[1]:
    #     x = x.T
    # x为观测向量矩阵，每列为一个观测向量
    # 观测向量个数n
    n = x.shape[1]
    # 以最后一列为参考计算协方差s，s的大小为(n-1,n-1)
    y = x[:, :-1] - x[:, -1][:, np.newaxis]
    s = np.cov(y, rowvar=False)
    dets = np.linalg.det(s)
    invs = np.linalg.inv(s)
    # 观测值的协方差矩阵，其中r1n,...,rnn为n个自由参数
    rcov = np.zeros((n, n), dtype=np.float64)
    # 确定n个自由参数[r1n,...,rnn]并初始化
    r0 = np.zeros(n, dtype=np.float64)
    u = np.ones((1, n - 1), dtype=np.float64)
    # 初始化rnn
    r0[-1] = 1.0 / (2.0 * u @ invs @ u.T)[0, 0]
    if r0.shape[0] - 1 != s.shape[0]:
        raise AttributeError("r.shape[0] - 1 != s.shape[0]")
    # bounds = []
    # for i in range(n - 1):
    #     bounds.append((None, None))
    # bounds.append((0, None))

    r = minimize(
        fun=_fun,
        x0=r0,
        args=(s, dets),
        method="SLSQP",
        tol=2e-10,
        constraints={"type": "ineq", "fun": _con, "args": (invs, dets)},
    )
    # breakpoint()
    # if not r.success:
    #     raise AttributeError('convergence: false')
    rcov[:, -1] = r.x
    rcov[-1, :] = r.x
    # rcov1 = np.copy(rcov)
    # for i in range(n - 1):
    #     for j in range(n - 1):
    #         # rij = sij + rin + rjn - rnn
    #         rcov[i, j] = s[i, j] + rcov[i, -1] + rcov[j, -1] - rcov[-1, -1]
    for i in range(n - 1):
        j = np.arange(n - 1)
        rcov[i, j] = s[i, j] + rcov[i, -1] + rcov[j, -1] - rcov[-1, -1]
    # print(np.sqrt(np.diag(rcov)))
    return np.sqrt(np.diag(rcov))


# from scipy.io import loadmat


def btch(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # if x.shape[0] < x.shape[1]:
    #     x = x.T
    std = gtch(x)
    var = std**2
    p = np.reciprocal(var) / np.reciprocal(var).sum()
    fusion = (x * p).sum(axis=1)
    return fusion.flatten(), std, p


def _fun(r: np.ndarray, s: np.ndarray, dets: float) -> float:
    # r为向量[r1n,...,rnn]
    nmin1 = s.shape[0]
    # 计算目标函数大小，目标函数为观测值协方差矩阵的上三角元素且不包括对角线元素的平方和
    # 根据协方差的双线性关系，可以求出rij与sij, rin, rjn, rnn的关系: rij = sij + rin + rjn - rnn
    f = 0.0
    for i in range(nmin1):
        f += r[i] ** 2
        for j in range(nmin1):
            if i < j:
                f += (s[i, j] + r[i] + r[j] - r[-1]) ** 2
    k = dets ** (2 / nmin1)
    f /= k
    return f


def _con(r: np.ndarray, invs: np.ndarray, dets: float) -> float:
    nmin1 = invs.shape[0]
    # r1为[[r1n,...,rn-1n]]
    r1 = r[:-1][np.newaxis, :]
    rnn = r[-1]
    # 计算约束函数，抄公式，如何推导没弄懂，-g<0，所以g>0
    g = (rnn - (r1 - rnn) @ invs @ (r1 - rnn).T) / dets ** (1 / nmin1)
    return g[0, 0]
