from pathlib import Path
import timeit

import numpy as np
import matplotlib.pyplot as plt

from shtoolkit.shcoeff import SpharmCoeff
from shtoolkit.shload import read_load_love_num
from shtoolkit.shspecial import standard
from shtoolkit.shtrans import fnALFs, spec2grd_fft, cilm2grid1, grid2cilm
from shtoolkit.shtrans import legendre, cilm2grid
# lat = np.linspace(90, -89, 180)
# lon = np.linspace(0, 359, 360)
# rad_colat = np.deg2rad(90-lat)
# rad_lon = np.deg2rad(lon)
# plm = fnALFs(rad_colat, 60)
# def trans():
#     # lat = np.linspace(90, -89, 180)
#     # lon = np.linspace(0, 359, 360)
#     # rad_colat = np.deg2rad(90-lat)
#     # rad_lon = np.deg2rad(lon)
#     # plm = fnALFs(rad_colat, 60)
#     ylms = gsm_post.coeffs[0, 0] - 1j*gsm_post.coeffs[0, 1]
#     delta_M = np.zeros((180, 61),dtype=np.complex128)
#     for k in range(0, 180):
#         # summation over all spherical harmonic degrees
#         delta_M[k] = np.sum(plm[k] * ylms, axis=0)
#     s = np.fft.ifft(delta_M, n=360, axis=1, norm='forward').real
#     # breakpoint()
#     return s

if __name__ == "__main__":
    # lmax = 60
    # slr_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\CSR_SLR_TN11E_TN11E.txt"
    # gsm_folder = Path("D:\\wjh_code\\TVG\\CSR\\unfilter")
    # gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"
    # deg1_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\TN-13_GEOC_CSR_RL0602.txt"
    # file1 = [slr_file1, slr_file1]

    # lln_file = "D:\\tvg_toolkit\\tvg_toolkit\\data\\lln_PREM.txt"
    # lln = read_load_love_num(lln_file, lmax)

    # gsm = SpharmCoeff.from_files(gsm_folder, lmax, "CSR GRACE")
    # gsm_post = gsm.rplce(["C20", "C30"], file1).corr_gia("ICE6G-D", gia_file1).rplce("DEG1", deg1_file).remove_mean_field().unitconvert('mmewh', lln)[0]

    # grd1 = gsm_post.expand(89, 60).data
    lmax = 60
    resol = 89
    rad_colat = np.linspace(0, np.pi, 2*(resol+1), endpoint=False)
    plm = legendre.fnALFs(rad_colat, lmax)
    grd = np.random.random((2*(resol+1), 4*(resol+1)))

    cilm = grid2cilm(grd, lmax)

    a = cilm2grid1(cilm, resol, lmax)
    b = cilm2grid.cilm2grid_fft(cilm, resol, lmax, plm)
    print(np.allclose(a, b))

    callable_object1 = lambda: cilm2grid1(cilm, resol, lmax)
    callable_object2 = lambda: cilm2grid.cilm2grid_integral(cilm, resol, lmax, plm)
    print(timeit.timeit(callable_object1, number=1000))
    print(timeit.timeit(callable_object2, number=1000))
    # breakpoint()