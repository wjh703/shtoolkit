import timeit

import numpy as np
from pyshtools.expand import SHExpandDH

from shtoolkit.shtrans import fnALFs, gridtocilm

if __name__ == "__main__":
    lmax = 60
    resol = 89
    rad_colat = np.linspace(0, np.pi, 2*(resol+1), endpoint=False)
    plm = fnALFs(rad_colat, lmax)
    # breakpoint()
    grd = np.random.random((2*(resol+1), 4*(resol+1)))

    a = SHExpandDH(grd, sampling=2, lmax_calc=lmax)
    b = gridtocilm(grd, lmax, plm)
    c = gridtocilm(grd, lmax, plm, 'integral')
    print(np.allclose(a, b))
    print(np.allclose(a, c))
    # breakpoint()
    callable_object1 = lambda: SHExpandDH(grd, sampling=2, lmax_calc=lmax)
    callable_object2 = lambda: gridtocilm(grd, lmax, plm)
    callable_object3 = lambda: gridtocilm(grd, lmax, plm, 'integral')
    print(timeit.timeit(callable_object1, number=1000))
    print(timeit.timeit(callable_object2, number=1000))
    print(timeit.timeit(callable_object3, number=1000))
    # breakpoint()