from typing import Literal

import numpy as np
from numpy.typing import NDArray


def helmert(A1: np.ndarray, P1: np.ndarray, L1: np.ndarray, A2: np.ndarray, P2: np.ndarray, L2: np.ndarray):
    n1 = len(L1)
    n2 = len(L2)
    N1 = A1.T * P1 @ A1
    W1 = A1.T * P1 @ L1
    while True:
        # for _ in range(1):
        N2 = A2.T * P2 @ A2
        N = N1 + N2
        invN = np.linalg.inv(N)

        W2 = A2.T * P2 @ L2
        W = W1 + W2

        x = invN @ W
        # x = np.linalg.solve(N, W)

        V1 = A1 @ x - L1
        V2 = A2 @ x - L2

        residual1 = V1.T * P1 @ V1
        residual2 = V2.T * P2 @ V2
        R = np.array([residual1, residual2])

        s1 = n1 - 2 * np.trace(invN @ N1) + np.trace(invN @ N1) ** 2
        s2 = np.trace(invN @ N1 @ invN @ N2)
        s3 = s2
        s4 = n2 - 2 * np.trace(invN @ N2) + np.trace(invN @ N2) ** 2

        S = np.array([[s1, s2], [s3, s4]])
        breakpoint()
        theta = np.linalg.solve(S, R)
        theta1 = round(theta[0], 4)
        theta2 = round(theta[1], 4)
        breakpoint()
        if theta1 != theta2:
            P2 *= theta1 / theta2
        else:
            break
        # rerr = np.abs(theta[1] - theta[0]) / (theta[0] + theta[1]) * 2
        # if rerr < 0.1:
        #     # if np.allclose(theta[0], theta[1], 0.05):
        #     break
        # else:
        #     P2 *= theta[0] / theta[1]

    return x


def vce(A1: np.ndarray, P1: np.ndarray, L1: np.ndarray, A2: np.ndarray, P2: np.ndarray, L2: np.ndarray):
    n1, n2 = len(L1), len(L2)

    var1, var2 = 1, 1

    iteration = 0
    while iteration < 100:
        N1 = (A1.T * P1 @ A1) / var1
        W1 = (A1.T * P1 @ L1) / var1

        N2 = (A2.T * P2 @ A2) / var2
        W2 = (A2.T * P2 @ L2) / var2
        # breakpoint()
        N = N1 + N2
        N += 0.025 * np.eye(N.shape[0])

        W = W1 + W2
        invN = np.linalg.inv(N)

        x = invN @ W

        v1 = A1 @ x - L1
        v2 = A2 @ x - L2

        # r1 = n1 - np.trace(invN @ N1)
        # r2 = n2 - np.trace(invN @ N2)

        r1 = n1 - np.trace(invN @ N1)
        r2 = n2 - np.trace(invN @ N2)

        var1_new = (v1.T * P1 @ v1) / r1
        var2_new = (v2.T * P2 @ v2) / r2

        var = np.array([var1, var2])
        var_new = np.array([var1_new, var2_new])

        # if np.linalg.norm(var - var_new) < 1e-5:
        #     break
        # if np.abs(1 - var1_new / var1) < 0.01 and np.abs(1 - var2_new / var2) < 0.01:
        #     break
        if np.isclose(var, var_new, rtol=0.01).all():
            break
        else:
            var1, var2 = var1_new, var2_new
            iteration += 1

    return x, var


def vce3(
    A1: np.ndarray,
    P1: np.ndarray,
    L1: np.ndarray,
    A2: np.ndarray,
    P2: np.ndarray,
    L2: np.ndarray,
    A3: np.ndarray,
    P3: np.ndarray,
    L3: np.ndarray,
):
    n1, n2, n3 = len(L1), len(L2), len(L1)

    # rms1 = np.sqrt((L1**2).sum())
    # rms2 = np.sqrt((L2**2).sum())
    # var1, var2 = rms1 / rms1, rms1 / rms2
    var1, var2, var3 = 1, 1, 1
    while True:
        N1 = (A1.T * P1 @ A1) / var1
        W1 = (A1.T * P1 @ L1) / var1

        N2 = (A2.T * P2 @ A2) / var2
        W2 = (A2.T * P2 @ L2) / var2

        N3 = (A3.T * P3 @ A3) / var3
        W3 = (A3.T * P3 @ L3) / var3

        N = N1 + N2 + N3
        W = W1 + W2 + W3

        invN = np.linalg.inv(N)

        x = invN @ W

        v1 = A1 @ x - L1
        v2 = A2 @ x - L2
        v3 = A3 @ x - L3

        r1 = np.trace(invN @ N1)
        r2 = np.trace(invN @ N2)
        r3 = np.trace(invN @ N3)

        var1_new = (v1.T * P1 @ v1) / (n1 - r1)
        var2_new = (v2.T * P2 @ v2) / (n2 - r2)
        var3_new = (v3.T * P3 @ v3) / (n3 - r3)

        var = np.array([var1, var2, var3])
        var_new = np.array([var1_new, var2_new, var3_new])
        norm = np.linalg.norm(var - var_new)
        if norm < 1e-5:
            break
        # # if any(var_new < 0):
        # #     breakpoint()
        # if np.allclose(var, var_new, rtol=0.01):
        #     break
        else:
            var1, var2, var3 = var1_new, var2_new, var3_new
    return x, var


def LS_VCE(A, L, Qt, sig0, eps):
    m = A.shape[0]
    # t = A.shape[1]
    p = int(Qt.shape[0] / m)
    # breakpoint()

    # 初始化QL
    QL = np.zeros((m, m))
    for i in range(p):
        QL += Qt[m * i : m * (i + 1), :]

    # 计算kx_LS
    PL = np.linalg.inv(QL)
    # PL = QL
    # PL[PL.nonzero()] = 1 / PL[PL.nonzero()]
    kx_LS = np.linalg.solve(A.T @ PL @ A, A.T @ PL @ L)
    # return kx_LS
    # 初始化迭代变量
    N = np.zeros((p, p))
    l = np.zeros((p, 1))
    S_sig = []

    cc = 0
    while True:
        cc += 1
        QL = np.zeros((m, m))
        for i in range(p):
            QL += sig0[i, 0] * Qt[m * i : m * (i + 1), :]

        PL = np.linalg.inv(QL)
        # PL = QL
        # PL[PL.nonzero()] = 1 / PL[PL.nonzero()]

        # 计算R和kx
        APA_inv = np.linalg.inv(A.T @ PL @ A)

        R = np.eye(m) - A @ APA_inv @ A.T @ PL
        kx = APA_inv @ A.T @ PL @ L
        e = L - A @ kx

        for i in range(p):
            QLi = Qt[m * i : m * (i + 1), :]
            for j in range(p):
                QLj = Qt[m * j : m * (j + 1), :]
                N[i, j] = 0.5 * np.trace(QLi @ PL @ R @ QLj @ PL @ R)
            # breakpoint()
            l[i, 0] = 0.5 * (e.T @ PL @ QLi @ PL @ e)[0, 0]

        sig = np.linalg.solve(N, l)
        S_sig.append(sig)

        # if np.linalg.norm(sig - sig0) < eps or cc > 50:
        if np.allclose(sig, sig0, rtol=0.001) or cc > 50:
            break

        sig0 = sig.copy()

    kx_VCE = kx
    Qsig = np.linalg.inv(N)
    S_sig = np.array(S_sig)

    return kx_LS, kx_VCE, S_sig, Qsig


REG_OPT = Literal["mytd", "mydd", "mysd"]
DTYPE = np.float64  # if you want to accelerate, you can use np.float32, but it may cause precision loss


def rvce(d: NDArray[DTYPE], reg: REG_OPT = "mytd") -> tuple[NDArray[DTYPE], NDArray[DTYPE]]:
    if reg not in ["mytd", "mydd", "mysd"]:
        raise ValueError(f"{reg} is not in {REG_OPT}")
    if d.ndim == 1:
        d = d[:, np.newaxis]
    if d.shape[0] < d.shape[1]:
        d = d.T

    md = d.shape[0]
    num = d.shape[1]
    if reg == "mytd":
        mx = md - 14
        D = np.zeros((mx, md), dtype=DTYPE)
        j = 0
        for i in range(mx):
            D[i, j], D[i, j + 1], D[i, j + 2] = 1, -2, 1
            D[i, j + 12], D[i, j + 13], D[i, j + 14] = -1, 2, -1
            j += 1
    elif reg == "mydd":
        mx = md - 13
        D = np.zeros((mx, md), dtype=DTYPE)
        j = 0
        for i in range(mx):
            D[i, j], D[i, j + 1] = 1, -1
            D[i, j + 12], D[i, j + 13] = -1, 1
            j += 1
    else:
        mx = md - 12
        D = np.zeros((mx, md), dtype=DTYPE)
        j = 0
        for i in range(mx):
            D[i, j], D[i, j + 12] = 1, -1
            j += 1

    # A = np.eye(md, dtype=DTYPE)
    A = np.tile(np.eye(md), (num, 1, 1))
    for i in range(d.shape[1]):
        gap_mask = d[:, i] == 0.0
        A[i, gap_mask, gap_mask] = 0.0

    R = D.T @ D
    vard = np.ones(num, dtype=DTYPE)
    varx = 1.0
    alpha = varx
    epsilon = 0.001
    chi = 3 * epsilon
    k = 0
    # start = time.time()
    while chi > epsilon and k < 10:
        re_vard = np.reciprocal(vard[:, np.newaxis, np.newaxis])
        re_varx = np.reciprocal(varx)
        N = np.sum(re_vard * A, axis=0) + re_varx * R  # inv(N)
        invN = np.linalg.inv(N)

        W = np.sum(re_vard.ravel() * d, axis=1)
        x = invN @ W

        tx = np.trace(re_varx * R @ invN)
        varx = np.reciprocal(mx - tx) * x @ R @ x
        # print(varx)
        # breakpoint()
        td = np.trace(re_vard * invN, axis1=1, axis2=2)
        # td = np.array([np.trace(i*invN) for i in np.reciprocal(vard)])
        vd = np.sqrt(md * np.reciprocal(md - td)) * (d - x[:, np.newaxis])
        vard = np.square(vd).sum(axis=0) / md
        # vard = np.array([np.square(vd[:, i]).sum() for i in range(num)])/md
        # print(vard)
        chi = np.abs((varx - alpha) / alpha, dtype=DTYPE)
        alpha = varx
        k += 1
    #     print(vard, '\n', varx)
    # breakpoint()
    stdd = np.sqrt(vard)
    # print(time.time() - start)
    return stdd, x


DTYPE = np.float64  # if you want to accelerate, you can use np.float32, but it may cause precision loss


def rvce_improve(
    d: NDArray[DTYPE], reg: Literal["mytd", "mydd", "mysd"] = "mytd", epsilon: float = 0.001
) -> tuple[NDArray[DTYPE], NDArray[DTYPE]]:
    if reg not in ["mytd", "mydd", "mysd"]:
        raise ValueError(f"Invalid regularization option: {reg}")
    if d.ndim == 1:
        d = d[:, np.newaxis]
    if d.shape[0] < d.shape[1]:
        d = d.T

    n = d.shape[0]  # number of unkonwn parameters/observations(missing data fill with 0)
    k = d.shape[1]  # number of datasets

    # generate coefficient matrix D for pseudo-observations
    if reg == "mytd":
        m = n - 14  # number of pseudo-observations
        D = np.zeros((m, n), dtype=DTYPE)
        j = 0
        for i in range(m):
            D[i, j], D[i, j + 1], D[i, j + 2] = 1, -2, 1
            D[i, j + 12], D[i, j + 13], D[i, j + 14] = -1, 2, -1
            j += 1
    elif reg == "mydd":
        m = n - 13  # number of pseudo-observations
        D = np.zeros((m, n), dtype=DTYPE)
        j = 0
        for i in range(m):
            D[i, j], D[i, j + 1] = 1, -1
            D[i, j + 12], D[i, j + 13] = -1, 1
            j += 1
    else:
        m = n - 12  # number of pseudo-observations
        D = np.zeros((m, n), dtype=DTYPE)
        j = 0
        for i in range(m):
            D[i, j], D[i, j + 12] = 1, -1
            j += 1

    # generate coefficient matrix A for each dataset
    A_matrices = []
    for i in range(k):
        nonzero_mask = d[:, i].nonzero()[0]
        A_matrices.append(np.eye(n, dtype=DTYPE)[nonzero_mask])

    # add pseudo-observations to coefficient matrix A
    A_matrices.append(D)

    # calculate variance components for each dataset and pseudo-observations iteratively
    # meanwhile generate regularized solution x
    variance_components = np.ones(k + 1, dtype=DTYPE)
    for _ in range(15):
        N = np.zeros((n, n))
        W = np.zeros(n)
        for i in range(k + 1):
            N += A_matrices[i].T @ A_matrices[i] / variance_components[i]
            if i < k:
                nonzero_mask = d[:, i].nonzero()[0]
                W += A_matrices[i].T @ d[nonzero_mask, i] / variance_components[i]
        invN = np.linalg.inv(N)
        x = invN @ W

        variance_components_new = np.zeros(k + 1)
        for i in range(k + 1):
            if i < k:
                nonzero_mask = d[:, i].nonzero()[0]
                v = A_matrices[i] @ x - d[nonzero_mask, i]  # residuals（改正数）
                observations = len(nonzero_mask)  # total observations
                r = np.trace(invN @ A_matrices[i].T @ A_matrices[i] / variance_components[i])  # redundant observations
                variance_components_new[i] = (v.T @ v) / (observations - r)
            else:
                v = A_matrices[i] @ x  # residuals（改正数）
                r = np.trace(invN @ A_matrices[i].T @ A_matrices[i] / variance_components[i])
                variance_components_new[i] = (v.T @ v) / (m - r)

        chi = np.abs((variance_components - variance_components_new) / variance_components, dtype=DTYPE)
        if chi.max() < epsilon:
            break
        else:
            variance_components = variance_components_new.copy()
    return variance_components**0.5, x
