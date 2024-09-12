# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.math cimport sqrt, sin, cos, pi
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as cnp

from .legendre import fnALF_cache
from .shfunc cimport cilm2vector

cnp.import_array()

cpdef cnp.ndarray[double, ndim=3] spharm_func(
        double lat,
        double lon, 
        int lmax
    ):
    cdef: 
        double ccos, ssin
        double factor = pi / 180
        double rad_colat = (90 - lat) * factor
        double rad_lon = lon * factor
        double[:,:] plm = fnALF_cache(rad_colat, lmax)
        double[:,:,:] yilm = np.zeros((2, lmax + 1, lmax + 1))
        Py_ssize_t l, m
    
    yilm[0] = plm
    yilm[1] = plm
    for m in range(lmax + 1):
        ccos = cos(m * rad_lon)
        ssin = sin(m * rad_lon)
        for l in range(m, lmax + 1):
            yilm[0, l, m] *= ccos
            yilm[1, l, m] *= ssin
    return np.asarray(yilm)


def spharm_func_map(
        double[:] lat,
        double[:] lon, 
        int lmax
    ):
    cdef: 
        int nlat = lat.shape[0]
        int nlon = lon.shape[0]
        int nvec = (lmax + 1) ** 2
        double[:,:,:] yilm
        cnp.ndarray[double, ndim=2] yilm_map = np.zeros((nlat * nlon, nvec))
        Py_ssize_t i, j
    
    for i in range(nlat):
        for j in range(nlon):
            yilm = spharm_func(lat[i], lon[j], lmax)
            yilm_map[i*nlon+j] = cilm2vector(yilm)

    fnALF_cache.cache_clear()

    return yilm_map