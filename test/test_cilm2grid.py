import timeit

import numpy as np
from pyshtools.expand import SHExpandDH, MakeGridDH

from shtoolkit.shtrans import fnALFs, cilmtogrid

if __name__ == "__main__":
    lmax = 60
    resol = 89
    rad_colat = np.linspace(0, np.pi, 2*(resol+1), endpoint=False)
    pilm = fnALFs(rad_colat, lmax)
    grd = np.random.random((2*(resol+1), 4*(resol+1)))

    cilm = SHExpandDH(grd, sampling=2, lmax_calc=lmax)
    a = MakeGridDH(cilm, sampling=2, lmax=resol, lmax_calc=lmax, extend=False)
    b = cilmtogrid(cilm, resol, lmax, pilm)
    print(np.allclose(a, b))

    callable_object1 = lambda: MakeGridDH(cilm, sampling=2, lmax=resol, lmax_calc=lmax, extend=False)
    callable_object2 = lambda: cilmtogrid(cilm, resol, lmax, pilm)
    print(timeit.timeit(callable_object1, number=1000))
    print(timeit.timeit(callable_object2, number=1000))
    # breakpoint()