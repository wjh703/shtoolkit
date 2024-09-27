from pathlib import Path

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from shtoolkit.shcoeffs import SpharmCoeff
from shtoolkit.shload import read_load_love_num
from shtoolkit.shspecial import standard


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
slr_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\GSFC_SLR_TN14.txt"
gsm_folder = "D:\\wjh_code\\TVG\\CSR\\unfilter"
gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"
gia_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\Purcell16.txt"
deg1_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\TN-13_GEOC_CSR_RL0602.txt"
file1 = [slr_file1, slr_file1]
file2 = [slr_file2, slr_file2]
lln_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\lln_PREM.txt"
lln = read_load_love_num(lln_file, lmax)

gsm = SpharmCoeff.from_files(gsm_folder, lmax)

rep_c20 = dict(rep="C20", file=slr_file1)
rep_c30 = dict(rep="C30", file=slr_file1)
rep_deg1 = dict(rep="DEG1", file=deg1_file)
b = gsm.replace([rep_c20, rep_c30]).corr_gia("ICE6G-D", gia_file1).remove_mean_field()  # type: ignore
c = gsm.replace(rep_deg1)  # type: ignore
c.coeffs -= c.coeffs[:100].mean(axis=0)
oc = np.loadtxt("D:\\tvg_toolkit\\masking\\data\\mask\\oceanmask\\ocean_buf300.txt")[:, 2].reshape(180, 360)
coeffs = standard(b.coeffs, b.unit, oc, lln, lmax, {"method": "buf_fs", "radius": 300}, mode="sal_rot")
coeffs = coeffs[:, *list(zip([0, 1, 0], [0, 1, 1], [1, 1, 1]))]
coeffs -= coeffs[:100].mean(axis=0)


deg1 = np.loadtxt(
    "D:/wjh_code/My_code/my_code_data/output/真实结果/CSR/GSM_like/buffer_300km.txt",
    delimiter=",",
)
deg1_time = np.copy(deg1[:, 0])
deg1 -= deg1[:100].mean(axis=0)

plt.plot(c.epochs, c.coeffs[:, 0, 1, 0])
plt.plot(deg1_time, deg1[:, 1])
plt.plot(b.epochs, coeffs[:, 0])
plt.plot(ldc_deg1[:, 0], ldc_deg1[:, 1])
plt.show()
plt.plot(c.epochs, c.coeffs[:, 0, 1, 1])
plt.plot(deg1_time, deg1[:, 2])
plt.plot(b.epochs, coeffs[:, 1])
plt.plot(ldc_deg1[:, 0], ldc_deg1[:, 2])
plt.show()
plt.plot(c.epochs, c.coeffs[:, 1, 1, 1])
plt.plot(deg1_time, deg1[:, 3])
plt.plot(b.epochs, coeffs[:, 2])
plt.plot(ldc_deg1[:, 0], ldc_deg1[:, 3])
plt.show()
