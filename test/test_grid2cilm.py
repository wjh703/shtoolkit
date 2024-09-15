import timeit

import numpy as np
from pyshtools.expand import SHExpandDH

from shtoolkit.shtrans import grid2cilm

if __name__ == "__main__":
    lmax = 60
    resol = 89
    rad_colat = np.linspace(0, np.pi, 2 * (resol + 1), endpoint=False)
    grd = np.random.random((2 * (resol + 1), 4 * (resol + 1)))

    a = SHExpandDH(grd, sampling=2, lmax_calc=lmax)
    b = grid2cilm(grd, lmax)
    c = grid2cilm(grd, lmax, mode="integral")
    print(np.allclose(a, b))
    print(np.allclose(a, c))
    del a, b, c
    print(f'lmax={lmax}, resol={resol}')
    callable_object1 = lambda: SHExpandDH(grd, sampling=2, lmax_calc=lmax)
    callable_object2 = lambda: grid2cilm(grd, lmax)
    callable_object3 = lambda: grid2cilm(grd, lmax, mode="integral")
    print(timeit.timeit(callable_object1, number=1000))
    print(timeit.timeit(callable_object2, number=1000))
    print(timeit.timeit(callable_object3, number=1000))
