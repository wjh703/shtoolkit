import time
import timeit

import numpy as np
import pyshtools as sh

from shtoolkit.shtrans import fnALFs, fnALF, spharm_func, spharm_func_map, vector2cilm

resol = 719
nlat = 2 * resol + 1
nlon = 2 * nlat

lat = np.linspace(90, -90, nlat, endpoint=False)
lon = np.linspace(0, 360, nlon, endpoint=False)
rad_colat = np.deg2rad(90 - lat)
rad_lon = np.deg2rad(lon)

lmax = 719

callable1 = lambda: fnALFs(rad_colat, lmax)
callable2 = lambda: fnALF(rad_colat[10], lmax)
print(timeit.timeit(callable1, number=1))
print(timeit.timeit(callable2, number=1))

# p1 = fnALFs(rad_colat, lmax)
# p2 = fnALF(rad_colat[10], lmax)
# print(np.allclose(p1[10], p2))

# y1 = spharm_func(lat[10], lon[10], lmax)
# y2 = sh.expand.spharm(lmax, 90-lat[10], lon[10])
# start = time.time()
# y3 = spharm_func_map(lat, lon, lmax)
# print(time.time()-start)
# breakpoint()
