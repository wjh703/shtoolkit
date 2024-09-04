from pathlib import Path
import timeit

import numpy as np
import matplotlib.pyplot as plt

from shtoolkit.shcoeff import SpharmCoeff
from shtoolkit.shload import read_load_love_num
from shtoolkit.shspecial import standard
from shtoolkit.shtrans import fnALFs, spec2grd_fft, cilm2grid1, grid2cilm1
from shtoolkit.shtrans import legendre, grid2cilm

def w1():
    w = []
    n = 2*(resol+1)
    for i in range(n):
        lat = rad_colat[i]
        s = 0
        for j in range(resol):
            s += np.sin((2*j+1)*lat)/(2*j+1)
        w.append(s*np.sin(np.pi*i/n)*8**0.5/n)
    return np.asarray(w)

def w2():
    w = []
    n = 2*(resol+1)
    for i in range(n):
        w.append(np.sin(rad_colat[i])*(np.pi/n))
    return np.asarray(w)

if __name__ == "__main__":

    lmax = 60
    resol = 89
    rad_colat = np.linspace(0, np.pi, 2*(resol+1), endpoint=False)

    plm = legendre.fnALFs(rad_colat, lmax)
    grd = np.random.random((2*(resol+1), 4*(resol+1)))


    a = grid2cilm1(grd, lmax)
    b = grid2cilm.grid2cilm_fft(grd, lmax, plm)
    c = grid2cilm.grid2cilm_integral(grd, lmax, plm)
    print(np.allclose(a, b))
    print(np.allclose(a, c))
    # breakpoint()
    callable_object1 = lambda: grid2cilm1(grd, lmax)
    callable_object2 = lambda: grid2cilm.grid2cilm_fft(grd, lmax, plm)
    callable_object3 = lambda: grid2cilm.grid2cilm_integral(grd, lmax, plm)
    print(timeit.timeit(callable_object1, number=1000))
    print(timeit.timeit(callable_object2, number=1000))
    print(timeit.timeit(callable_object3, number=1000))
    # breakpoint()