import time
import timeit

import numpy as np
import pyshtools as sh

from shtoolkit.shtrans import fnALFs, fnALF, fnALFs_cache

resol = 89
nlat = 2 * resol + 1
nlon = 2 * nlat

lat = np.linspace(90, -90, nlat, endpoint=False)
lon = np.linspace(0, 360, nlon, endpoint=False)
rad_colat = np.deg2rad(90 - lat)
rad_lon = np.deg2rad(lon)

lmax = 89

callable1 = lambda: fnALFs_cache(tuple(rad_colat), lmax)
callable2 = lambda: np.asarray(fnALFs_cache(tuple(rad_colat), lmax).transpose(1, 2, 0), order='C')
# callable2 = lambda: fnALF(rad_colat[10], lmax)
print(timeit.timeit(callable1, number=100))
print(timeit.timeit(callable2, number=100))

# p1 = fnALFs(rad_colat, lmax)
# p2 = fnALF(rad_colat[10], lmax)
# print(np.allclose(p1[10], p2))

# y1 = spharm_func(lat[10], lon[10], lmax)
# y2 = sh.expand.spharm(lmax, 90-lat[10], lon[10])
# start = time.time()
# y3 = spharm_func_map(lat, lon, lmax)
# print(time.time()-start)
# breakpoint()
