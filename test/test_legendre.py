import time
import timeit

import numpy as np
import pyshtools as sh

from shtoolkit.shtrans import fnALFs, fnALF, fnALFs_cache, fnALFs_refined

resol = 89
nlat = 2 * resol + 1
nlon = 2 * nlat

lat = np.linspace(90, -90, nlat, endpoint=False)
lon = np.linspace(0, 360, nlon, endpoint=False)
rad_colat = np.deg2rad(90 - lat)[1:91]
rad_lon = np.deg2rad(lon)

lmax = 60
a = fnALFs(rad_colat, lmax)
b = fnALFs_refined(rad_colat, lmax)
c = sh.legendre.legendre(1000, np.cos(rad_colat[1]))
print(np.allclose(a, b))

callable1 = lambda: fnALFs(rad_colat, lmax)
callable2 = lambda: fnALFs_refined(rad_colat, lmax)
# callable2 = lambda: fnALF(rad_colat[10], lmax)
print(timeit.timeit(callable1, number=100))
print(timeit.timeit(callable2, number=100))
