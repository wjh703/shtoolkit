from pathlib import Path

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from shtoolkit.shcoeffs import SpharmCoeff
from shtoolkit.shread import read_load_love_num, read_slr_5x5
from shtoolkit.shspecial.grace_obp import standard, combination
from shtoolkit.shtime import grace_time, vce

def vce2(A1, P1, L1, A2, P2, L2, max_iter=100, tol=1e-5):
        """
        改进的方差分量估计函数
        返回:
        sigma: 方差分量估计 [grace_var, slr_var]
        C: 融合参数的协方差矩阵
        """
        sigma = np.array([1.0, 2.0])  # 初始值
        C = np.eye(A1.shape[1])  # 初始化协方差矩阵

        for _ in range(max_iter):
            # 构建正则化法方程
            N1 = (A1.T @ P1 @ A1) / sigma[0]
            N2 = (A2.T @ P2 @ A2) / sigma[1]
            N = N1 + N2 +   np.eye(N1.shape[0])  # 正则化项

            # 求解参数
            try:
                x = np.linalg.solve(N, (A1.T @ P1 @ L1) / sigma[0] + (A2.T @ P2 @ L2) / sigma[1])
                C = np.linalg.inv(N)  # 参数协方差矩阵
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(N, (A1.T @ P1 @ L1) / sigma[0] + (A2.T @ P2 @ L2) / sigma[1], rcond=1e-6)[0]
                C = np.linalg.pinv(N)

            # 残差计算
            v_grace = (A1 @ x - L1).flatten()
            v_slr = (A2 @ x - L2).flatten()

            # 方差更新
            sigma_new = np.array([
                max((v_grace @ P1 @ v_grace) / 3, 1e-15),  # 保证非负
                max((v_slr @ P2 @ v_slr) / 3, 1e-15)
            ])

            if np.allclose(sigma, sigma_new, rtol=tol):
                break
            sigma = sigma_new

        return sigma, C

slr_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\CSR_SLR_TN11E_TN11E.txt"
slr_file2 = "D:\\tvg_toolkit\\shtoolkit\\src\\shtoolkit\\data\\TN-14_C30_C20_GSFC_SLR.txt"
gsm_folder = "D:\\wjh_code\\TVG\\CSR\\unfilter"
gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"

lln_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\lln_PREM.txt"
lln = read_load_love_num(lln_file, 60)

slr_5x5_epochs, slr_5x5_coeffs = read_slr_5x5('D:/tvg_toolkit/shtoolkit/src/shtoolkit/data/gsfc_slr_5x5c61s61.txt')
slr_gsm = SpharmCoeff(slr_5x5_coeffs, slr_5x5_epochs, "stokes").resample(grace_time("200204", "202212")).corr_gia("ICE6G-D", gia_file1).remove_mean_field()
slr_gsm.coeffs[:, :, 6, 1:] = 0
slr_gsm.coeffs -= slr_gsm.coeffs[:100].mean(axis=0)



rep_c20 = dict(name="C20", file=slr_file2)
rep_c30 = dict(name="C30", file=slr_file2)
grace_gsm = SpharmCoeff.from_files(gsm_folder, 6).replace([rep_c20, rep_c30]).corr_gia("ICE6G-D", gia_file1).remove_mean_field()  # type: ignore
grace_gsm.coeffs -= grace_gsm.coeffs[:100].mean(axis=0)

oc1 = np.loadtxt("D:\\tvg_toolkit\\masking\\data\\mask\\oceanmask\\ocean_buf0.txt")[:, 2].reshape(180, 360)
oc2 = np.loadtxt("D:\\tvg_toolkit\\masking\\data\\mask\\oceanmask\\ocean_buf300.txt")[:, 2].reshape(180, 360)
# comb_coeffs = combination(grace_gsm.coeffs, slr_gsm.coeffs, oc2, 'stokes', lln, lln)
slr_coeffs = standard(slr_gsm.coeffs, slr_gsm.unit, oc1, lln, 6, {"method": "FM_gs", "radius": 2000}, mode="sal_rot")
grace_coeffs = standard(grace_gsm.coeffs, grace_gsm.unit, oc2, lln, 60, {"method": "buf", "radius": None}, mode="sal_rot")

slr_deg1 = slr_coeffs[:, [0, 0, 1], [1, 1, 1], [0, 1, 1]] * 3 **.5 * 6371000 * 1000
grace_deg1 = grace_coeffs[:, [0, 0, 1], [1, 1, 1], [0, 1, 1]] * 3 **.5 * 6371000 * 1000

A = np.eye(3)  # 设计矩阵
P_grace = np.ones(3)
P_slr = np.ones(3)
comb_coeffs = []
var = []
for i in range(216):
    comb, v1 = vce(A, P_grace, grace_deg1[i], A, P_slr, slr_deg1[i])
    var.append(v1)
    comb_coeffs.append(comb)
var = np.array(var)
comb_coeffs = np.array(comb_coeffs)

# var = np.array(var)
# fig, ax = plt.subplots()
# ax.plot(var[:, 0])
# ax.plot(var[:, 1])
# plt.show()



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8), sharex=True)
axes[0].plot(slr_gsm.epochs, slr_coeffs[:, 0, 1, 0], label="SLR")
axes[0].plot(grace_gsm.epochs, grace_coeffs[:, 0, 1, 0], label="GRACE")
axes[0].plot(grace_gsm.epochs, comb_coeffs[:, 0], label="COMB")
axes[1].plot(slr_gsm.epochs, slr_coeffs[:, 0, 1, 1])
axes[1].plot(grace_gsm.epochs, grace_coeffs[:, 0, 1, 1])
axes[1].plot(grace_gsm.epochs, comb_coeffs[:, 1])
axes[2].plot(slr_gsm.epochs, slr_coeffs[:, 1, 1, 1])
axes[2].plot(grace_gsm.epochs, grace_coeffs[:, 1, 1, 1])
axes[2].plot(grace_gsm.epochs, comb_coeffs[:, 2])
axes[0].legend()
plt.tight_layout()
plt.show()





