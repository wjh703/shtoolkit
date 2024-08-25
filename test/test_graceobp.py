from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from shtoolkit.shcoeff import SpharmCoeff
from shtoolkit.shload import read_load_love_num
from shtoolkit.shspecial import standard

lmax = 60
slr_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\CSR_SLR_TN11E_TN11E.txt"
slr_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\GSFC_SLR_TN14.txt"
gsm_folder = Path("D:\\wjh_code\\TVG\\CSR\\unfilter")
gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"
gia_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\Purcell16.txt"
deg1_file = "D:\\tvg_toolkit\\tvg_toolkit1\\data\\TN-13_GEOC_CSR_RL0602.txt"
file1 = [slr_file1, slr_file1]
file2 = [slr_file2, slr_file2]
lln_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\lln_PREM.txt"
lln = read_load_love_num(lln_file, lmax)

a = SpharmCoeff.from_files(gsm_folder, lmax, "CSR GRACE")
b = a.rplce(["C20", "C30"], file1).corr_gia("ICE6G-D", gia_file1).remove_mean_field()

c = a.rplce(["C20", "C30"], file2).corr_gia("ICE6G-C", gia_file2).rplce("DEG1", deg1_file)
c.coeffs -= c.coeffs[:100].mean(axis=0)
oc = np.loadtxt("D:\\tvg_toolkit\\tvg_toolkit\\data\\oc_func_300km.txt")[:, 2].reshape(180, 360)
coeffs = standard(b.coeffs, b.unit, oc, lln, lmax, {"method": "FM_fs", "radius": 300}, mode="sal")
coeffs = coeffs[:, *list(zip([0, 1, 0], [0, 1, 1], [1, 1, 1]))]
coeffs -= coeffs[:100].mean(axis=0)
deg1 = np.loadtxt(
    "D:/wjh_code/My_code/my_code_data/output/真实结果/CSR/GSM_like/buffer_300km.txt",
    delimiter=",",
)
deg1_time = np.copy(deg1[:, 0])
deg1 -= deg1[:100].mean(axis=0)
# b = a.rplce('C20', slr_file1).rplce('C30', slr_file1).corr_gia('ICE6G-D', gia_file1)
# c = a.rplce('C20', slr_file2).rplce('C30', slr_file2).corr_gia('ICE6G-C', gia_file2)
# breakpoint()
plt.plot(c.epochs, c.coeffs[:, 0, 1, 0])
plt.plot(deg1_time, deg1[:, 1])
plt.plot(b.epochs, coeffs[:, 0])
plt.show()
plt.plot(c.epochs, c.coeffs[:, 0, 1, 1])
plt.plot(deg1_time, deg1[:, 2])
plt.plot(b.epochs, coeffs[:, 1])
plt.show()
plt.plot(c.epochs, c.coeffs[:, 1, 1, 1])
plt.plot(deg1_time, deg1[:, 3])
plt.plot(b.epochs, coeffs[:, 2])
plt.show()
