import numpy as np
from numba import jit

__all__ = ["lstsq_map", "cosine_fitting", "sine_fitting"]


@jit(nopython=True)
def lstsq_map(
    dtime: np.ndarray,
    x: np.ndarray,
    reference_time: float = 2002.0027,
):

    n, nrow, ncol = x.shape

    amp1 = np.zeros((nrow, ncol))
    amp1_std = np.zeros((nrow, ncol))
    phi1 = np.zeros((nrow, ncol))
    phi1_std = np.zeros((nrow, ncol))
    amp2 = np.zeros((nrow, ncol))
    amp2_std = np.zeros((nrow, ncol))
    phi2 = np.zeros((nrow, ncol))
    phi2_std = np.zeros((nrow, ncol))
    k = np.zeros((nrow, ncol))
    k_std = np.zeros((nrow, ncol))
    linear = np.zeros((n, nrow, ncol))
    residual = np.zeros((n, nrow, ncol))

    for i in range(nrow):
        for j in range(ncol):
            (
                amp1[i, j],
                amp1_std[i, j],
                phi1[i, j],
                phi1_std[i, j],
                amp2[i, j],
                amp2_std[i, j],
                phi2[i, j],
                phi2_std[i, j],
                k[i, j],
                k_std[i, j],
                linear[:, i, j],
                residual[:, i, j],
            ) = cosine_fitting(dtime, x[:, i, j], reference_time)

    return (
        amp1,
        amp1_std,
        phi1,
        phi1_std,
        amp2,
        amp2_std,
        phi2,
        phi2_std,
        k,
        k_std,
        linear,
        residual,
    )


def sine_fitting(
    dtime: np.ndarray, data: np.ndarray, reference_time: float = 2002.0027
) -> tuple[
    float, float, float, float, float, float, float, float, float, float, np.ndarray, np.ndarray
]:
    """fitting a1sin(2pit+phi) + a2sin(4pit+phi) + kt + b by the least square"""
    if dtime.size != data.size:
        raise ValueError(f"{dtime.size} != {data.size}")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    t = dtime - reference_time
    omega = 2.0 * np.pi

    # 观测值向量矩阵, 每列代表一个观测向量
    # 列向量分别为： (1, t, cos(2pi*t), sin(2pi*t), cos(2pi*2t), sin(2pi*2t))
    # 观测方程： bx = data， x = (btb)-1(btdata)
    b = np.c_[
        np.ones_like(t),
        t,
        np.cos(omega * t),
        np.sin(omega * t),
        np.cos(omega * 2.0 * t),
        np.sin(omega * 2.0 * t),
    ]

    # 计算BTB
    btb = b.T @ b
    # 计算协因数矩阵
    qxx = np.linalg.inv(btb)
    # 利用最小二乘求x
    w = qxx @ b.T
    xx = w @ data

    # 2维转1维
    x = xx.flatten()

    # x[2] -> sin(phi1)
    # x[3] -> cos(phi1)
    amp1 = np.sqrt(x[2] ** 2 + x[3] ** 2)
    if x[2] > 0.0:  # sin(phi1) > 0 [0, pi/2], [pi/2, pi]
        phi1 = np.arccos(x[3] / amp1)
    elif x[2] < 0.0:  # sin(phi1) < 0 [pi, 3*pi/2], [3*pi/2, 2*pi]
        phi1 = omega - np.arccos(x[3] / amp1)
    elif x[2] == 0.0 and x[3] == 1.0:
        phi1 = 0.0
    elif x[2] == 0.0 and x[3] == -1.0:
        phi1 = np.pi
    elif x[2] == 0.0 and x[3] == 0.0:
        phi1 = 0.0
    else:
        raise ValueError("Unknown Error!")
    phi1 = np.rad2deg(phi1)

    # x[4] -> sin(phi2)
    # x[5] -> cos(phi2)
    amp2 = np.sqrt(x[4] ** 2 + x[5] ** 2)
    if x[4] > 0.0:
        phi2 = np.arccos(x[5] / amp2)
    elif x[4] < 0.0:
        phi2 = omega - np.arccos(x[5] / amp2)
    elif x[4] == 0.0 and x[5] == 1.0:
        phi2 = 0.0
    elif x[4] == 0.0 and x[5] == -1.0:
        phi2 = np.pi
    elif x[4] == 0.0 and x[5] == 0.0:
        phi2 = 0.0
    else:
        raise ValueError("Unknown Error!")
    phi2 = np.rad2deg(phi2)

    # 计算残差
    res = data - b @ xx

    # 计算单位权方差
    uni_var = res.T @ res / (len(t) - 6)
    # 观测值方差=单位权方差*协因数矩阵
    obs_var = uni_var * qxx

    linear = x[1] * t + x[0]
    k = x[1]
    # 2倍标准差，表示95.4%的置信区间
    k_std = 2.0 * np.sqrt(obs_var[1, 1])
    # 根据协方差传播率计算振幅和相位的2倍标准差
    amp1_var = (x[2] ** 2 * obs_var[2, 2] + x[3] ** 2 * obs_var[3, 3]) / (amp1**2.0)
    amp1_std = 2.0 * np.sqrt(amp1_var)
    if np.abs(x[2] / x[3]) <= 1:
        x_23 = x[2] / x[3]  # 表示tan
        # 将反正切函数泰勒展开，再对x_23求全微分
        phi1_var = (1 - x_23**2 + x_23**4 - x_23**6 + x_23**8 - x_23**10) ** 2 * (
            (1 / x[3]) ** 2 * obs_var[2, 2] + (x[2] / x[3] ** 2) ** 2 * obs_var[3, 3]
        )
        phi1_std = np.rad2deg(2.0 * np.sqrt(phi1_var))
    else:
        x_32 = x[3] / x[2]  # 表示cot
        # 将反余切函数泰勒展开，再对x_32求全微分
        phi1_var = (-1 + x_32**2 - x_32**4 + x_32**6 - x_32**8 + x_32**10) ** 2 * (
            (x[3] / x[2] ** 2) ** 2 * obs_var[2, 2] + (1 / x[2]) ** 2 * obs_var[3, 3]
        )
        phi1_std = np.rad2deg(2.0 * np.sqrt(phi1_var))

    amp2_var = (x[4] ** 2 * obs_var[4, 4] + x[5] ** 2 * obs_var[5, 5]) / (amp2**2.0)
    amp2_std = 2.0 * np.sqrt(amp2_var)
    if np.abs(x[4] / x[5]) <= 1:
        x_45 = x[4] / x[5]
        phi2_var = (1 - x_45**2 + x_45**4 - x_45**6 + x_45**8 - x_45**10) ** 2 * (
            (1 / x[5]) ** 2 * obs_var[4, 4] + (x[4] / x[5] ** 2) ** 2 * obs_var[5, 5]
        )
        phi2_std = np.rad2deg(2.0 * np.sqrt(phi2_var))
    else:
        x_54 = x[5] / x[4]
        phi2_var = (-1 + x_54**2 - x_54**4 + x_54**6 - x_54**8 + x_54**10) ** 2 * (
            (x[5] / x[4] ** 2) ** 2 * obs_var[4, 4] + (1 / x[4]) ** 2 * obs_var[5, 5]
        )
        phi2_std = np.rad2deg(2.0 * np.sqrt(phi2_var))

    return (
        amp1,
        amp1_std,
        phi1,
        phi1_std,
        amp2,
        amp2_std,
        phi2,
        phi2_std,
        k,
        k_std,
        linear,
        res.flatten(),
    )


@jit(nopython=True, cache=True)
def cosine_fitting(
    dtime: np.ndarray, data: np.ndarray, reference_time: float = 2002.0014
) -> tuple[
    float, float, float, float, float, float, float, float, float, float, np.ndarray, np.ndarray
]:
    """fitting a1cos(2pit-phi) + a2cos(4pit-phi) + kt + b by the least square"""
    if dtime.size != data.size:
        raise ValueError(f"{dtime.size} != {data.size}")

    t = dtime - reference_time
    omega = 2.0 * np.pi
    d = np.copy(data)
    # 观测值向量矩阵, 每列代表一个观测向量
    # 列向量分别为： (1, t, cos(2pi*t), sin(2pi*t), cos(2pi*2t), sin(2pi*2t))
    b = np.column_stack(
        (
            np.ones_like(t),
            t,
            np.cos(omega * t),
            np.sin(omega * t),
            np.cos(omega * 2.0 * t),
            np.sin(omega * 2.0 * t),
        )
    )

    # 计算BTB
    btb = b.T @ b
    # 计算协因数矩阵
    qxx = np.linalg.inv(btb)
    # 利用最小二乘求x
    x = qxx @ b.T @ d
    # 2维转1维, (6,1) -> (6,)
    # x[2] -> cos(phi1)
    # x[3] -> sin(phi1)
    amp1 = np.sqrt(x[2] ** 2 + x[3] ** 2)
    if x[3] > 0.0:  # sin(phi1) > 0 [0, pi/2], [pi/2, pi]
        phi1 = np.arccos(x[2] / amp1)
    elif x[3] < 0.0:  # sin(phi1) < 0 [pi, 3*pi/2], [3*pi/2, 2*pi]
        phi1 = omega - np.arccos(x[2] / amp1)
    elif x[3] == 0.0 and x[2] == 1.0:
        phi1 = 0.0
    elif x[3] == 0.0 and x[2] == -1.0:
        phi1 = np.pi
    elif x[3] == 0.0 and x[2] == 0.0:
        phi1 = 0.0
    else:
        raise ValueError("Unknown Error!")
    phi1 = np.rad2deg(phi1)

    # x[4] -> cos(phi2)
    # x[5] -> sin(phi2)
    amp2 = np.sqrt(x[4] ** 2 + x[5] ** 2)
    if x[5] > 0.0:
        phi2 = np.arccos(x[4] / amp2)
    elif x[5] < 0.0:
        phi2 = omega - np.arccos(x[4] / amp2)
    elif x[5] == 0.0 and x[4] == 1.0:
        phi2 = 0.0
    elif x[5] == 0.0 and x[4] == -1.0:
        phi2 = np.pi
    elif x[5] == 0.0 and x[4] == 0.0:
        phi2 = 0.0
    else:
        raise ValueError("Unknown Error!")
    phi2 = np.rad2deg(phi2)

    # 计算残差
    res = data - b @ x
    # 计算单位权方差
    uni_var = res.T @ res / (len(t) - 6)
    # 观测值方差=单位权方差*协因数矩阵
    obs_var = uni_var * qxx

    linear = x[1] * t + x[0]
    k = x[1]
    # 2倍标准差，表示95.4%的置信区间
    k_std = 2.0 * np.sqrt(obs_var[1, 1])
    # 根据协方差传播率计算振幅和相位的2倍标准差
    amp1_var = (x[2] ** 2 * obs_var[2, 2] + x[3] ** 2 * obs_var[3, 3]) / (amp1**2.0)
    amp1_std = 2.0 * np.sqrt(amp1_var)
    # x3/x2 = sin/cos = tan9
    # x2/x3 = cos/sin = cot
    if np.abs(x[3] / x[2]) <= 1:
        x_32 = x[3] / x[2]  # 表示tan
        # 将反正切函数泰勒展开，再对x_32求全微分
        phi1_var = (1 - x_32**2 + x_32**4 - x_32**6 + x_32**8 - x_32**10) ** 2 * (
            (1 / x[2]) ** 2 * obs_var[3, 3] + (x[3] / x[2] ** 2) ** 2 * obs_var[2, 2]
        )
        phi1_std = np.rad2deg(2.0 * np.sqrt(phi1_var))
    else:
        x_23 = x[2] / x[3]  # 表示cot
        # 将反余切函数泰勒展开，再对x_23求全微分
        phi1_var = (-1 + x_23**2 - x_23**4 + x_23**6 - x_23**8 + x_23**10) ** 2 * (
            (x[2] / x[3] ** 2) ** 2 * obs_var[3, 3] + (1 / x[3]) ** 2 * obs_var[2, 2]
        )
        phi1_std = np.rad2deg(2.0 * np.sqrt(phi1_var))

    amp2_var = (x[4] ** 2 * obs_var[4, 4] + x[5] ** 2 * obs_var[5, 5]) / (amp2**2.0)
    amp2_std = 2.0 * np.sqrt(amp2_var)
    if np.abs(x[5] / x[4]) <= 1:
        x_54 = x[5] / x[4]
        phi2_var = (1 - x_54**2 + x_54**4 - x_54**6 + x_54**8 - x_54**10) ** 2 * (
            (1 / x[4]) ** 2 * obs_var[5, 5] + (x[5] / x[4] ** 2) ** 2 * obs_var[4, 4]
        )
        phi2_std = np.rad2deg(2.0 * np.sqrt(phi2_var))
    else:
        x_45 = x[4] / x[5]
        phi2_var = (-1 + x_45**2 - x_45**4 + x_45**6 - x_45**8 + x_45**10) ** 2 * (
            (x[4] / x[5] ** 2) ** 2 * obs_var[5, 5] + (1 / x[5]) ** 2 * obs_var[4, 4]
        )
        phi2_std = np.rad2deg(2.0 * np.sqrt(phi2_var))

    return amp1, amp1_std, phi1, phi1_std, amp2, amp2_std, phi2, phi2_std, k, k_std, linear, res
