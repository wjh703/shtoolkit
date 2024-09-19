# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.math cimport sin, cos, pi

import numpy as np
import scipy as sp

from .legendre import fnALFs_cache

"""
Reference
---------
[1] Wieczorek, M. A., & Meschede, M. (2018). SHTools: Tools for working 
        with spherical harmonics. Geochemistry, Geophysics, Geosystems, 
        19, 2574-2592. https://doi.org/10.1029/2018GC007529
"""


def cilm2grid_fft(
        double[:,:,:] cilm,
        int resol,
        int calc_lmax = -1
    ):

    cdef: 
        int nlat = 2 * (resol + 1)
        int nlon = 2 * nlat
        int lmax
        tuple rad_colat = tuple(np.linspace(0, pi, nlat, endpoint=False))
        double[:,:,:] pilm
        double complex[:,:] fcoef
        Py_ssize_t k, l, m
        double am, bm

    if nlat % 2 != 0:
        raise ValueError(f"Invalid value of nlat: {nlat}, (expected even)")

    lmax = cilm.shape[1]-1
    if calc_lmax < 0 or calc_lmax > lmax:
        calc_lmax = lmax
    
    pilm = fnALFs_cache(rad_colat, calc_lmax)
    
    fcoef = np.zeros((nlat, calc_lmax + 1), dtype=np.complex128)
    for k in range(nlat):
        for m in range(calc_lmax + 1):
            am = 0.0
            bm = 0.0
            for l in range(m, calc_lmax + 1):
                am += cilm[0, l, m] * pilm[k, l, m]
                bm += cilm[1, l, m] * pilm[k, l, m]
            if m:
                fcoef[k, m] = (am - 1j * bm) / 2
            else:
                fcoef[k, m] = am - 1j * bm

    return sp.fft.irfft(fcoef, nlon, norm='forward')
 

def cilm2grid_fft_refined(
        double[:,:,:] cilm,
        int resol,
        int calc_lmax = -1
    ):

    cdef: 
        int nlat = 2 * (resol + 1)
        int nlon = 2 * nlat
        int lmax
        tuple rad_colat = tuple(np.linspace(0, pi, nlat, endpoint=False))
        double[:,:,:] pilm
        double[:,:] plm, plms
        double complex[:,:] lat_fft
        Py_ssize_t k, l, m
        double am, bm
        double ams, bms

    if nlat % 2 != 0:
        raise ValueError(f"Invalid value of nlat: {nlat}, (expected even)")

    lmax = cilm.shape[1]-1
    if calc_lmax < 0 or calc_lmax > lmax:
        calc_lmax = lmax
    
    pilm = fnALFs_cache(rad_colat, calc_lmax)
    
    lat_fft = np.empty((nlat, calc_lmax + 1), dtype=np.complex128)

    for k in range(nlat // 2):
        ks = nlat - 1 - k

        plm = pilm[k]
        plms = pilm[ks]

        for m in range(calc_lmax + 1):
            am = 0.0
            bm = 0.0
            ams = 0.0
            bms = 0.0
            for l in range(m, calc_lmax + 1):
                am += cilm[0, l, m] * plm[l, m]
                bm += cilm[1, l, m] * plm[l, m]
                ams += cilm[0, l, m] * plms[l, m]
                bms += cilm[1, l, m] * plms[l, m]
            if m:
                lat_fft[k, m] = (am - 1j * bm) / 2
                lat_fft[ks, m] = (ams - 1j * bms) / 2
            else:
                lat_fft[k, m] = am - 1j * bm
                lat_fft[ks, m] = ams - 1j * bms

    return sp.fft.irfft(lat_fft, nlon, norm='forward')


def cilm2grid_integral(
        double[:,:,:] cilm,
        int resol,
        int calc_lmax = -1
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
        tuple rad_colat = tuple(np.linspace(0, pi, nlat, endpoint=False))
        double[:] rad_lon = np.linspace(0, 2 * pi, nlon, endpoint=False)
        double[:,:,:] pilm
    
    if nlat % 2 != 0:
        raise ValueError(f"Invalid value of nlat: {nlat}, (expected even)")

    lmax = cilm.shape[1] - 1
    if calc_lmax < 0 or calc_lmax > lmax:
        calc_lmax = lmax
    
    pilm = fnALFs_cache(rad_colat, calc_lmax)

    am = np.zeros((nlat, calc_lmax + 1))
    bm = np.zeros((nlat, calc_lmax + 1))
    for k in range(nlat):
        for m in range(calc_lmax + 1):
            a = 0.0
            b = 0.0
            for l in range(m, calc_lmax + 1):
                a += cilm[0, l, m] * pilm[k, l, m]
                b += cilm[1, l, m] * pilm[k, l, m]
            am[k, m] = a
            bm[k, m] = b

    ccos = np.zeros((calc_lmax + 1, nlon))
    ssin = np.zeros((calc_lmax + 1, nlon))
    for l in range(calc_lmax+1):
        for k in range(nlon):
            ccos[l, k] = cos(l * rad_lon[k])
            ssin[l, k] = sin(l * rad_lon[k])

    return np.dot(am, ccos) + np.dot(bm, ssin)