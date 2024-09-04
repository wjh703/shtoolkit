# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.math cimport sin, cos, pi

import numpy as np
import scipy

from .legendre cimport fnALFs

def cilm2grid_fft(
        double[:,:,:] cilm,
        int resol,
        int calc_lmax = -1,
        double[:,:,:] pilm = None
    ):

    cdef: 
        int nlat = 2*(resol+1)
        int nlon = 2*nlat
        int lmax
        double complex[:,:] fcoef
        Py_ssize_t k, l, m
        double am, bm

    lmax = cilm.shape[1]-1
    if calc_lmax < 0 or calc_lmax > lmax:
        calc_lmax = lmax
    
    if pilm is None:
        pilm = fnALFs(np.linspace(0, pi, nlat, endpoint=False), calc_lmax)
        
    if pilm.shape[0] != nlat:
        raise ValueError(f"The dimension-1 value of 'pilm' is unequal to 'nlat'")
    
    fcoef = np.zeros((nlat, calc_lmax+1), dtype=np.complex128)
    for k in range(nlat):
        for m in range(calc_lmax+1):
            am = 0.0
            bm = 0.0
            for l in range(m, calc_lmax+1):
                am += cilm[0, l, m] * pilm[k, l, m]
                bm += cilm[1, l, m] * pilm[k, l, m]
            fcoef[k, m] = am-1j*bm
    return scipy.fft.ifft(fcoef, nlon, axis=1, norm='forward').real


def cilm2grid_integral(
        double[:,:,:] cilm,
        int resol,
        int calc_lmax = -1,
        double[:,:,:] pilm = None
    ):

    cdef: 
        int nlat = 2 * (resol + 1)
        int nlon = 2 * nlat
        double[:,:] am
        double[:,:] bm
        Py_ssize_t k, l, m
        double[:,:] ccos
        double[:,:] ssin
        double a, b
        double[:] rad_colat = np.linspace(0, pi, nlat, endpoint=False)
        double[:] rad_lon = np.linspace(0, 2*pi, nlon, endpoint=False)

    lmax = cilm.shape[1] - 1

    if calc_lmax < 0 or calc_lmax > lmax:
        calc_lmax = lmax
    
    if pilm is None:
        pilm = fnALFs(rad_colat, calc_lmax)
        
    if pilm.shape[0] != nlat:
        raise ValueError(f"The dimension-1 value of 'pilm' is unequal to 'nlat'")

    am = np.zeros((nlat, calc_lmax+1))
    bm = np.zeros((nlat, calc_lmax+1))
    for k in range(nlat):
        for m in range(calc_lmax+1):
            a = 0.0
            b = 0.0
            for l in range(m, calc_lmax+1):
                a += cilm[0, l, m] * pilm[k, l, m]
                b += cilm[1, l, m] * pilm[k, l, m]
            am[k, m] = a
            bm[k, m] = b

    ccos = np.zeros((calc_lmax+1, nlon))
    ssin = np.zeros((calc_lmax+1, nlon))
    for l in range(calc_lmax+1):
        for k in range(nlon):
            ccos[l, k] = cos(l * rad_lon[k])
            ssin[l, k] = sin(l * rad_lon[k])

    return np.dot(am, ccos) + np.dot(bm, ssin)