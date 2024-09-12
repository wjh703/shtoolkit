import time
import timeit
import numpy as np

from shtoolkit.shtrans import fnALFs, calc_yilm_mat, fnALF, fnALFs1


lat = np.linspace(90, -89, 180)
lon = np.linspace(0, 359, 360)
rad_colat = np.deg2rad(90 - lat)
rad_lon = np.deg2rad(lon)

lmax = 60

p1 = fnALFs(rad_colat, lmax)
p2 = fnALF(rad_colat[10], lmax)
p3 = fnALFs1(rad_colat, lmax)
breakpoint()
callable_obj1 = lambda: np.linspace(0, np.pi, 360, endpoint=False)
print(timeit.timeit(callable_obj1, number=1000))
start = time.time()
p2 = fnALFs(rad_colat, lmax)
print(time.time() - start)

# start = time.time()
# ylm = calc_yilm_mat(lat, lon, lmax)
# print(time.time() - start)
