from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from shtoolkit import shunit
from shtoolkit.shcoeff import SpharmCoeff
from shtoolkit.shload import read_load_love_num

slr_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\CSR_SLR_TN11E_TN11E.txt"
slr_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\GSFC_SLR_TN14.txt"
gsm_folder = Path("D:\\wjh_code\\TVG\\CSR\\unfilter")
gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"
gia_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\Purcell16.txt"
deg1_file = "D:\\tvg_toolkit\\tvg_toolkit1\\data\\TN-13_GEOC_CSR_RL0602.txt"
lln_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\lln_PREM.txt"
file1 = [slr_file1, slr_file1]
file2 = [slr_file2, slr_file2]

lln = read_load_love_num(lln_file, 60)
gsm = SpharmCoeff.from_files(gsm_folder, 60, "CSR GRACE")
gsm_post = (
    gsm.rplce(["C20", "C30"], file1)
    .corr_gia("ICE6G-D", gia_file2)
    .rplce("DEG1", deg1_file)
    .remove_mean_field()
)
print(gsm_post.name)
gsm_post_convert = gsm_post.unitconvert("kgm2mass", lln)
a = shunit.convert(gsm_post.coeffs, "kgm2mass", "mgeo", lln)
# c = gsm.rplce(['C20', 'C30'], file2).corr_gia('ICE6G-C', gia_file2)
plt.plot(gsm_post.epochs, gsm_post_convert.coeffs[:, 0, 2, 0])
plt.show()
plt.plot(gsm_post.epochs, gsm_post_convert.coeffs[:, 0, 3, 0])
plt.show()
