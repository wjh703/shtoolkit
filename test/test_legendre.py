import time
import timeit

import numpy as np
import pyshtools as sh

from shtoolkit.shtrans import fnALFs, fnALF, spharm_func, spharm_func_map, vector2cilm, calc_yilm_mat


lat = np.linspace(90, -89, 180)
lon = np.linspace(0, 359, 360)
rad_colat = np.deg2rad(90 - lat)
rad_lon = np.deg2rad(lon)

lmax = 60

p1 = fnALFs(rad_colat, lmax)
p2 = fnALF(rad_colat[10], lmax)
print(np.allclose(p1[10], p2))

y1 = spharm_func(lat[10], lon[10], lmax)
y2 = sh.expand.spharm(lmax, 90-lat[10], lon[10])
start = time.time()
y3 = spharm_func_map(lat, lon, lmax)
print(time.time()-start)
breakpoint()
start = time.time()
y4 = calc_yilm_mat(lat, lon, lmax)
print(time.time()-start)
breakpoint()
#print(np.allclose(y3, y4))
# print(np.allclose(vector2cilm(y3[10*360+10]), y1))

# start = time.time()
# ylm = calc_yilm_mat(lat, lon, lmax)
# print(time.time() - start)
