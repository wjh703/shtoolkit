# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.math cimport sin, cos, pi, sqrt
from libc.stdlib cimport malloc, free

import numpy as np
import scipy

from .legendre cimport fnALFs

def grid2cilm_fft(
    double[:,:] grid,
    int calc_lmax = -1,
    double[:,:,:] pilm = None
):
    cdef: 
        int nlat = grid.shape[0]
        int nlon = grid.shape[1]
        int resol = nlat / 2 - 1
        double c, s
        double fpi = 4.0 * pi
        Py_ssize_t k, l, m
        double[:] rad_colat = np.linspace(0, pi, nlat, endpoint=False)
        double *weight
        double complex[:,:] lat_fft
        double[:,:,:] cilm


    if calc_lmax < 0:
        calc_lmax = resol
    elif calc_lmax > resol:
        raise ValueError(f"Invalid value of calc_lmax: {calc_lmax}, must smaller than 'resol': {2*nlat-1}")

    if pilm is None:
        pilm = fnALFs(rad_colat, calc_lmax)
        
    if pilm.shape[0] != nlat:
        raise ValueError(f"The dimension-1 value of 'pilm' is unequal to 'nlat'")

    weight = weight_dh(nlat)
    lat_fft = scipy.fft.rfft(grid, axis=1)[:, :calc_lmax+1]
    cilm = np.zeros((2, calc_lmax+1, calc_lmax+1))
    for l in range(calc_lmax+1):
        for m in range(l+1):
            c = 0.0
            s = 0.0
            for k in range(nlat):
                c += pilm[k, l, m] * lat_fft[k, m].real * weight[k]
                if m:
                    s += pilm[k, l, m] * (-lat_fft[k, m].imag) * weight[k]
            cilm[0, l, m] = c / fpi
            cilm[1, l, m] = s / fpi
    free(weight)
    return np.asarray(cilm)


def grid2cilm_integral(
    double[:,:] grid,
    int calc_lmax = -1,
    double[:,:,:] pilm = None
):
    cdef: 
        int nlat = grid.shape[0] 
        int nlon = grid.shape[1]
        double dlat = pi / nlat
        double dlon = 2.0 * pi / nlon
        int resol = nlat / 2 - 1
        Py_ssize_t k, l, m
        double[:] rad_colat = np.linspace(0, pi, nlat, endpoint=False)
        double[:] rad_lon = np.linspace(0, 2.0 * pi, nlon, endpoint=False)
        double *weight = <double *> malloc(sizeof(double) * nlat)
        # double *weight = weight_dh(nlat)
        double[:,:] ccos
        double[:,:] ssin
        double[:,:] am
        double[:,:] bm
        double c, s
        double fpi = 4.0 * pi
        double[:,:,:] cilm

    if calc_lmax < 0:
        calc_lmax = resol
    elif calc_lmax > resol:
        raise ValueError(f"Invalid value of calc_lmax: {calc_lmax}, must smaller than 'resol': {2*nlat-1}")

    if pilm is None:
        pilm = fnALFs(rad_colat, calc_lmax)
        
    if pilm.shape[0] != nlat:
        raise ValueError(f"The dimension-1 value of 'pilm' is unequal to 'nlat'")

    for k in range(nlat):
       weight[k] = sin(rad_colat[k]) * dlat * dlon 

    ccos = np.zeros((nlon, calc_lmax+1))
    ssin = np.zeros((nlon, calc_lmax+1))
    for k in range(nlon):
        for m in range(calc_lmax+1):
            ccos[k, m] = cos(rad_lon[k] * m)
            ssin[k, m] = sin(rad_lon[k] * m)

    am = np.dot(grid, ccos)
    bm = np.dot(grid, ssin)
    cilm = np.zeros((2, calc_lmax+1, calc_lmax+1))
    for l in range(calc_lmax+1):
        for m in range(l+1):
            c = 0.0
            s = 0.0
            for k in range(nlat):
                c += pilm[k, l, m] * am[k, m] * weight[k]
                if m:
                    s += pilm[k, l, m] * bm[k, m] * weight[k]
            cilm[0, l, m] = c/fpi
            cilm[1, l, m] = s/fpi
    free(weight)
    return np.asarray(cilm)


cdef inline double *weight_dh(int nlat):
    cdef:
        int lmax = nlat // 2 -1 
        double *w = <double *> malloc(sizeof(double) * nlat)
        Py_ssize_t j, l
        double s, colat
        double dlon = pi / nlat

    for j in range(nlat):
        s = 0
        colat = pi * j / nlat
        for l in range(lmax + 1):
            s += sin((2 * l + 1) * colat) / (2 * l + 1)
        w[j] = s * sin(colat) * sqrt(8.0) / nlat * sqrt(2.0) * dlon
    return w