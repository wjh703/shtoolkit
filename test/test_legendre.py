import time

import numpy as np

from shtoolkit.shtrans import fnalf, fnALFs, vector2cilm, cilm2vector, calc_yilm_mat


lat = np.linspace(90, -89, 180)
lon = np.linspace(0, 359, 360)
rad_colat = np.deg2rad(90-lat)
rad_lon = np.deg2rad(lon)

lmax = 60

start = time.time()
p1 = fnalf(rad_colat, lmax)
print(time.time()-start)
start = time.time()
p2 = fnALFs(rad_colat, lmax)
print(time.time()-start)

start = time.time()
ylm = calc_yilm_mat(lat, lon, lmax)
print(time.time()-start)
breakpoint()

