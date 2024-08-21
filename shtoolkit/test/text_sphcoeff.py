from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from shtoolkit.shcoeff import SpharmCoeff

slr_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\CSR_SLR_TN11E_TN11E.txt"
slr_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\GSFC_SLR_TN14.txt"
gsm_folder = Path("D:\\wjh_code\\TVG\\CSR\\unfilter")
gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"
gia_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\Purcell16.txt"
deg1_file = "D:\\tvg_toolkit\\tvg_toolkit1\\data\\TN-13_GEOC_CSR_RL0602.txt"
file1 = [slr_file1, slr_file1]
file2 = [slr_file2, slr_file2]
a = SpharmCoeff.from_files(gsm_folder, 60, "CSR GRACE")
b = (
    a.rplce(["C20", "C30"], file1)
    .corr_gia("ICE6G-D", gia_file1)
    .rplce("DEG1", deg1_file)
)
print(b.name)
c = a.rplce(["C20", "C30"], file2).corr_gia("ICE6G-C", gia_file2)
d = b.remove_mean_field().smooth("gs")
e = d.expand(89)
f = e.expand(60)
print(np.allclose(d.coeffs, f.coeffs))

# b = a.rplce('C20', slr_file1).rplce('C30', slr_file1).corr_gia('ICE6G-D', gia_file1)
# c = a.rplce('C20', slr_file2).rplce('C30', slr_file2).corr_gia('ICE6G-C', gia_file2)
# breakpoint()
plt.plot(b.epochs, b.coeffs[:, 0, 1, 0])
plt.show()
plt.plot(b.epochs, b.coeffs[:, 0, 1, 1])
plt.show()
plt.plot(b.epochs, b.coeffs[:, 1, 1, 1])
plt.show()
plt.plot(b.epochs, b.coeffs[:, 0, 2, 0])
plt.plot(c.epochs, c.coeffs[:, 0, 2, 0])
plt.show()
plt.plot(a.epochs, a.coeffs[:, 0, 3, 0])
plt.plot(b.epochs, b.coeffs[:, 0, 3, 0])
plt.plot(c.epochs, c.coeffs[:, 0, 3, 0])
plt.show()
