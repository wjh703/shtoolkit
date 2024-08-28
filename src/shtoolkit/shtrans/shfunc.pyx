# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
'''# cython: initializedcheck=False'''

from libc.math cimport sqrt, sin, cos
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef inline int plmidx(int degree, int order) except -1:
    cdef int idx
    idx = (degree * (degree + 1)) / 2 + order
    return idx

cpdef cnp.ndarray[double, ndim=3] fnALFs(double[:] rad_colat, int lmax):
    cdef:
        int vecnum = plmidx(lmax, lmax) + 1
        double *al = <double *> malloc(sizeof(double) * (lmax+1))
        double *bl = <double *> malloc(sizeof(double) * (lmax+1))
        double *clm = <double *> malloc(sizeof(double) * vecnum)
        double *dlm = <double *> malloc(sizeof(double) * vecnum)
        double *elm = <double *> malloc(sizeof(double) * vecnum)
        Py_ssize_t l, m
        int vecidx

    for l in range(2, lmax+1):
        al[l] = sqrt(<double>(2 * l + 1) / (2 * l - 1))
        bl[l] = sqrt(<double>(2 * (l - 1) * (2 * l + 1)) / (l * (2 * l - 1)))
        for m in range(l+1):
            vecidx = plmidx(l, m)
            clm[vecidx] =  sqrt(<double>((2 * l + 1) * (l + m) * (l - m)) / (2 * l - 1)) / l
            dlm[vecidx] = sqrt(<double>((2 * l + 1) * (l - m - 1) * (l - m)) / (2 * l - 1)) / (2 * l)
            if m-1 == 0:
                elm[vecidx] = sqrt(<double>(2 * (2 * l + 1) * (l + m - 1) * (l + m)) / ((2 - 1) * (2 * l - 1))) / (2 * l)
            else:
                elm[vecidx] = sqrt(<double>(2 * (2 * l + 1) * (l + m - 1) * (l + m)) / ((2 - 0) * (2 * l - 1))) / (2 * l)

    cdef: 
        int len_colat = rad_colat.shape[0]
        double[:,:,:] pilm = np.zeros((len_colat, lmax+1, lmax+1))
        double t, u
        Py_ssize_t i
    
    for i in range(len_colat):
        t = cos(rad_colat[i])
        u = sin(rad_colat[i])
        pilm[i, 0, 0] = 1.0
        pilm[i, 1, 0] = sqrt(3) * t
        pilm[i, 1, 1] = sqrt(3) * u
        for l in range(2, lmax + 1):
            for m in range(l + 1):
                vecidx = plmidx(l, m)
                if m == 0:
                    pilm[i, l, m] = al[l]*t*pilm[i, l-1, 0]-bl[l]*(u/2)*pilm[i, l-1, 1]
                elif m == lmax:
                    pilm[i, l, m] = u*elm[vecidx]*pilm[i, l-1, m-1]
                else:
                    pilm[i, l, m] = clm[vecidx]*t*pilm[i, l-1, m] - u*(dlm[vecidx]*pilm[i, l-1, m+1]-elm[vecidx]*pilm[i, l-1, m-1])
    free(al)
    free(bl)
    free(clm)
    free(dlm)
    free(elm)
    return np.asarray(pilm)


cdef cnp.ndarray[double, ndim=2] calc_yilm_mat(
        cnp.ndarray[double, ndim=1] lat,
        cnp.ndarray[double, ndim=1] lon, 
        int lmax
    ):
    cdef: 
        int nlat = len(lat)
        int nlon = len(lon)
        int nvec = (lmax + 1) ** 2
        double ccos, ssin
        double[:] rad_colat = np.deg2rad(90 - lat)
        double[:] rad_lon = np.deg2rad(lon)
        double[:,:,:] pilm = fnALFs(rad_colat, lmax)
        double[:,:,:] yilm = np.zeros((2, lmax + 1, lmax + 1))
        double[:,:] yilm_mat = np.zeros((nlat * nlon, nvec))
        Py_ssize_t i, j, m, k
    
    for i in range(nlat):
        for j in range(nlon):
            yilm[0] = pilm[i]
            yilm[1] = pilm[i]
            for m in range(lmax+1):
                ccos = cos(m*rad_lon[j])
                ssin = sin(m*rad_lon[j])
                for k in range(m, lmax+1):
                    yilm[0, k, m] *= ccos
                    yilm[1, k, m] *= ssin
            yilm_mat[i*nlon+j] = cilm2vector(yilm)
    return np.asarray(yilm_mat)


cpdef cnp.ndarray[double, ndim=1] cilm2vector(double[:,:,:] coeffs):
    cdef: 
        int lmax = coeffs.shape[1]-1
        int cilm_num = (lmax+1)**2
        double[:] vector = np.zeros(cilm_num)
        int k = 0
        Py_ssize_t l, m

    for l in range(lmax+1):
        for m in range(l+1):
            vector[k] = coeffs[0, l, m]
            k += 1
        for m in range(1, l+1):
            vector[k] = coeffs[1, l, m]
            k += 1

    return np.asarray(vector)


cpdef cnp.ndarray[double, ndim=3] vector2cilm(double[:] vector):
    cdef: 
        int vector_len = len(vector)
        int lmax = <int>sqrt(vector_len) - 1
        double[:,:,:] coeffs = np.zeros((2, lmax+1, lmax+1))
        int i = 1
        int l = 0
        int m = 0
        Py_ssize_t k

    coeffs[0, 0, 0] = vector[0]
    for k in range(1, vector_len):
        m += 1
        if m > l and i == 0:
            i = 1
            m = 1
        elif m > l and i == 1:
            l += 1
            m = 0
            i = 0
        coeffs[i, l, m] = vector[k]

    return np.asarray(coeffs)


def r2c(double[:,:,:] cilm):
    cdef:
        int lmax = cilm.shape[1]-1
        double[:,:,:] cilm_complex = np.zeros_like(cilm)
        double sqrt2 = sqrt(2.0)
        Py_ssize_t i, j
    
    for i in range(lmax+1):
        for j in range(i+1):
            if j == 0:
                cilm_complex[0, i, j] = cilm[0, i, j]
            else:
                cilm_complex[0, i, j] = cilm[0, i, j] / sqrt2
                cilm_complex[1, i, j] = - cilm[1, i, j] / sqrt2
    
    return np.asarray(cilm_complex)


def c2r(double[:,:,:] cilm_complex):
    cdef:
        int lmax = cilm_complex.shape[1]-1
        double[:,:,:] cilm = np.zeros_like(cilm_complex)
        double sqrt2 = sqrt(2.0)
        Py_ssize_t i, j
    
    for i in range(lmax+1):
        for j in range(i+1):
            if j == 0:
                cilm[0, i, j] = cilm_complex[0, i, j]
            else:
                cilm[0, i, j] = cilm_complex[0, i, j] * sqrt2
                cilm[1, i, j] = - cilm_complex[1, i, j] * sqrt2
    
    return np.asarray(cilm)