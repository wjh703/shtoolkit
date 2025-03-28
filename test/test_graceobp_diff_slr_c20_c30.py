from pathlib import Path

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 

from shtoolkit.shcoeffs import SpharmCoeff
from shtoolkit.shread import read_load_love_num, read_technical_note_deg1
from shtoolkit.shspecial.grace_obp import standard


def load_ldc():
    import re
    from astropy.time import Time
    from shtoolkit.shtime import date_to_decimal_year

    ldc = loadmat("C:\\Users\\huan\\Desktop\\基金本子\\参考文献\\LDCmgm90_20200108_GSM_GAA_GAB_GAC_GAD.mat")
    ldc_clm = ldc["LDCmgm_GSM_Cnm"]
    ldc_slm = ldc["LDCmgm_GSM_Snm"]
    ldc_epochs = ldc["LDCmgm_Epoch"].ravel()
    iso = Time(ldc_epochs, format="mjd").iso
    t = []
    for s in iso:
        dt = "".join(re.findall(r"(\d{4})-(\d{2})-(\d{2})", s)[0])
        t.append(date_to_decimal_year(dt))

    # ldc_deg = ldc['LDCmgm_Degree']
    # ldc_ord = ldc['LDCmgm_Order']
    # ldc_ind = np.hstack((ldc_deg, ldc_ord))
    ldc_deg1 = np.c_[ldc_clm[1], ldc_clm[2], ldc_slm[2]]
    ldc_deg1 -= ldc_deg1[:100].mean(axis=0)
    return np.hstack((np.asarray(t)[:, np.newaxis], ldc_deg1))

ldc_deg1 = load_ldc()
lmax = 60

slr_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\CSR_SLR_TN11E_TN11E.txt"
slr_file2 = "D:\\tvg_toolkit\\shtoolkit\\src\\shtoolkit\\data\\TN-14_C30_C20_GSFC_SLR.txt"
gsm_folder = "D:\\wjh_code\\TVG\\CSR\\unfilter"
gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"
gia_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\Purcell16.txt"
deg1_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\TN-13_GEOC_CSR_RL0602.txt"

deg1_epochs, deg1_coeffs, _ = read_technical_note_deg1(deg1_file)
deg1_coeffs -= deg1_coeffs[:100].mean(axis=0)

lln_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\lln_PREM.txt"
lln = read_load_love_num(lln_file, lmax)

gsm1 = SpharmCoeff.from_files(gsm_folder, lmax)
gsm2 = gsm1.copy()

rep_c20 = dict(name="C20", file=slr_file2)
rep_c30 = dict(name="C30", file=slr_file2)
rep_gsfc = [dict(name='C20', file=slr_file2), dict(name='C30', file=slr_file2)]
rep_csr = [dict(name='C20', file=slr_file1), dict(name='C30', file=slr_file1)]

rep_deg1 = dict(name="DEG1", file=deg1_file)
gsm1_processed = gsm1.replace(rep_csr).corr_gia("ICE6G-D", gia_file1).remove_mean_field()  # type: ignore
gsm2_processed = gsm2.replace(rep_gsfc).corr_gia("ICE6G-D", gia_file1).remove_mean_field()  # type: ignore


oc = np.loadtxt("D:\\tvg_toolkit\\masking\\data\\mask\\oceanmask\\ocean_buf300.txt")[:, 2].reshape(180, 360)

coeffs1 = standard(gsm1_processed.coeffs, gsm1_processed.unit, oc, lln, lmax, {"method": "buf_fs", "radius": 50}, mode="sal_rot")
coeffs2 = standard(gsm2_processed.coeffs, gsm2_processed.unit, oc, lln, lmax, {"method": "buf_fs", "radius": 50}, mode="sal_rot")

coeffs1 = coeffs1[:, *list(zip([0, 1, 0], [0, 1, 1], [1, 1, 1]))]
coeffs2 = coeffs2[:, *list(zip([0, 1, 0], [0, 1, 1], [1, 1, 1]))]
coeffs1 -= coeffs1[:100].mean(axis=0)
coeffs2 -= coeffs2[:100].mean(axis=0)

correlations = np.corrcoef(coeffs1.T, coeffs2.T)
print(correlations)

# coeffs = standard(b.coeffs, b.unit, oc, lln, lmax, {"method": "buf_fs", "radius": 300}, mode="sal_rot")

# coeffs = coeffs[:, *list(zip([0, 1, 0], [0, 1, 1], [1, 1, 1]))]
# coeffs -= coeffs[:100].mean(axis=0)


deg1 = np.loadtxt(
    "D:/wjh_code/My_code/my_code_data/output/真实结果/CSR/GSM_like/buffer_300km.txt",
    delimiter=",",
)
# deg1 = np.loadtxt(
#     "D:/wjh_code/My_code/my_code_data/output/真实结果/finally/CSR_FM_MPIOM_RL07.txt",
#     delimiter=",",
# )

deg1_time = np.copy(deg1[:, 0])
deg1 -= deg1[:100].mean(axis=0)

dtime = gsm1_processed.epochs
dtime1 = dtime[dtime<2017.5]
dtime2 = dtime[dtime>=2017.5]
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex='col')
axes[0].plot(dtime1, coeffs1[dtime<2017.5, 0], label="With CSR C20 solutions", c='r')
axes[0].plot(dtime2, coeffs1[dtime>=2017.5, 0], c='r')
axes[0].plot(dtime1, coeffs2[dtime<2017.5, 0], label="With GSFC C20 solutions", ls=(0, (5, 5)), c='b')
axes[0].plot(dtime2, coeffs2[dtime>=2017.5, 0], ls=(0, (5, 5)), c='b')
axes[0].legend()
# plt.plot(deg1_time, deg1[:, 1])
# plt.plot(b.epochs, coeffs[:, 0])
# plt.plot(ldc_deg1[:, 0], ldc_deg1[:, 1])
axes[1].plot(dtime1, coeffs1[dtime<2017.5, 1], c='r')
axes[1].plot(dtime2, coeffs1[dtime>=2017.5, 1], c='r')
axes[1].plot(dtime1, coeffs2[dtime<2017.5, 1], ls=(0, (5, 5)), c='b')
axes[1].plot(dtime2, coeffs2[dtime>=2017.5, 1], ls=(0, (5, 5)), c='b')
# plt.plot(deg1_time, deg1[:, 2])
# plt.plot(b.epochs, coeffs[:, 1])
# plt.plot(ldc_deg1[:, 0], ldc_deg1[:, 2])
axes[2].plot(dtime1, coeffs1[dtime<2017.5, 2], c='r')
axes[2].plot(dtime2, coeffs1[dtime>=2017.5, 2], c='r')
axes[2].plot(dtime1, coeffs2[dtime<2017.5, 2], ls=(0, (5, 5)), c='b')
axes[2].plot(dtime2, coeffs2[dtime>=2017.5, 2], ls=(0, (5, 5)), c='b')
# plt.plot(deg1_time, deg1[:, 3])
# plt.plot(b.epochs, coeffs[:, 2])
# plt.plot(ldc_deg1[:, 0], ldc_deg1[:, 3])
axes[0].grid(True, linestyle='--', alpha=0.7, which='both')
axes[1].grid(True, linestyle='--', alpha=0.7, which='both')
axes[2].grid(True, linestyle='--', alpha=0.7, which='both')
# axes[0].set_title('Geocenter Motion')
# axes[0].set_ylabel('Z', rotation=0)
# axes[1].set_ylabel('X', rotation=0)
# axes[2].set_ylabel('Y', rotation=0)
axes[0].spines[:].set_linewidth(1)
axes[1].spines[:].set_linewidth(1)
axes[2].spines[:].set_linewidth(1)
axes[0].set_ylabel(r'$C_{10}$', rotation=0)
axes[1].set_ylabel(r'$C_{11}$', rotation=0)
axes[2].set_ylabel(r'$S_{11}$', rotation=0)
# axes[0].set_ylim(-8, 8)
# axes[1].set_ylim(-6, 6)
# axes[2].set_ylim(-6, 6)
# axes[0].set_yticks(np.linspace(-8, 8, 5))
# axes[1].set_yticks(np.linspace(-6, 6, 5))
# axes[2].set_yticks(np.linspace(-6, 6, 5))
axes[0].set_xticks(np.linspace(2002, 2022, 6))
# axes[1, 1].xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(2004, 2016, 4)))
axes[0].xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(2004, 2024, 6)))
# plt.suptitle('Geocenter Motion')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.04, hspace=0.12)
plt.savefig('D:/wjh_code/My_code/my_code_data/output/using_different_C2O_solutions.png', dpi=600, bbox_inches="tight")
plt.show()

