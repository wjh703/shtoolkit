import timeit

import numpy as np
# from pyshtools.expand import SHExpandDH, MakeGridDH
# import scipy.fft

from shtoolkit.shtrans import fnALFs, cilm2grid, grid2cilm

if __name__ == "__main__":
    lmax = 359
    resol = 359
    rad_colat = np.linspace(0, np.pi, 2 * (resol + 1), endpoint=False)
    pilm = fnALFs(rad_colat, lmax)
    grd = np.random.random((2 * (resol + 1), 4 * (resol + 1)))

    # cilm = SHExpandDH(grd, sampling=2, lmax_calc=lmax)
    # a = MakeGridDH(cilm, sampling=2, lmax=resol, lmax_calc=lmax, extend=False)
    cilm = grid2cilm(grd, lmax)
    b = cilm2grid(cilm, resol, lmax)
    c = cilm2grid(cilm, resol, lmax, mode="integral")
    # breakpoint()
    # breakpoint()
    # print(np.allclose(grd, a))
    print(np.allclose(c, b))
    print(np.allclose(grd, c))
    # callable_object1 = lambda: MakeGridDH(cilm, sampling=2, lmax=resol, lmax_calc=lmax, extend=False)
    callable_object2 = lambda: cilm2grid(cilm, resol, lmax)
    callable_object3 = lambda: cilm2grid(cilm, resol, lmax, mode="integral")
    print(f"lmax:{lmax}, resol:{resol}")
    # print(timeit.timeit(callable_object1, number=100))
    print(timeit.timeit(callable_object2, number=10))
    print(timeit.timeit(callable_object3, number=10))
