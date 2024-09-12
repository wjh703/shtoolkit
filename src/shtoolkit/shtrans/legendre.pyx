# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

from libc.math cimport sqrt, sin, cos
from libc.stdlib cimport malloc, free
from cython cimport floating

from functools import cache
import numpy as np
cimport numpy as cnp

cnp.import_array()

"""
Reference
---------
[1] Xing, Z., Li, S., Tian, M. et al. Numerical experiments on column-wise recurrence 
        formula to compute fully normalized associated Legendre functions of ultra-high 
        degree and order. J Geod 94, 2 (2020). https://doi.org/10.1007/s00190-019-01331-0
"""

@cache
def fnALFs_cache(tuple rad_colat, int lmax):
    return fnALFs(np.asarray(rad_colat), lmax)


@cache
def fnALF_cache(floating rad_colat, int lmax):
    return fnALF(rad_colat, lmax)


cdef inline int plmidx(int degree, int order) except -1:
    cdef int idx
    idx = (degree * (degree + 1)) / 2 + order
    return idx


cdef class RecursiveCoef:
    cdef:
        int lmax
        int nvec
        double *al
        double *bl
        double *clm
        double *dlm
        double *elm

    def __cinit__(self, int lmax):
        cdef int nvec = plmidx(lmax, lmax) + 1
        self.al = <double *> malloc(sizeof(double) * (lmax + 1))
        self.bl = <double *> malloc(sizeof(double) * (lmax + 1))
        self.clm = <double *> malloc(sizeof(double) * nvec)
        self.dlm = <double *> malloc(sizeof(double) * nvec)
        self.elm = <double *> malloc(sizeof(double) * nvec) 
        if (self.al == NULL or self.bl == NULL or self.clm == NULL 
            or self.dlm == NULL or self.elm == NULL):
            raise MemoryError('allocate memory failed')
    
    def __init__(self, int lmax):
        self.lmax = lmax

    cdef void compute(self):
        cdef:
            int ivec
            Py_ssize_t l, m

        for l in range(self.lmax + 1):
            self.al[l] = sqrt(<double>(2 * l + 1) / (2 * l - 1))
            self.bl[l] = sqrt(<double>(2 * (l - 1) * (2 * l + 1)) / (l * (2 * l - 1)))
            for m in range(l + 1):
                ivec = plmidx(l, m)
                self.clm[ivec] =  sqrt(<double>((2 * l + 1) * (l + m) * (l - m)) / (2 * l - 1)) / l
                self.dlm[ivec] = sqrt(<double>((2 * l + 1) * (l - m - 1) * (l - m)) / (2 * l - 1)) / (2 * l)
                if m - 1 == 0:
                    self.elm[ivec] = sqrt(<double>(2 * (2 * l + 1) * (l + m - 1) * (l + m)) / ((2 - 1) * (2 * l - 1))) / (2 * l)
                else:
                    self.elm[ivec] = sqrt(<double>(2 * (2 * l + 1) * (l + m - 1) * (l + m)) / ((2 - 0) * (2 * l - 1))) / (2 * l)
    
    def __dealloc__(self):
        free(self.al)
        free(self.bl)
        free(self.clm)
        free(self.dlm)
        free(self.elm)


cpdef cnp.ndarray[double, ndim=2] fnALF(double rad_colat, int lmax):
    cdef:
        RecursiveCoef rc = RecursiveCoef(lmax)
        double[:,:] plm = np.zeros((lmax + 1, lmax + 1))
        double t, u
        Py_ssize_t l, m
        int ivec

    rc.compute()
    t = cos(rad_colat)
    u = sin(rad_colat)
    plm[0, 0] = 1.0
    plm[1, 0] = sqrt(3.0) * t
    plm[1, 1] = sqrt(3.0) * u
    for l in range(2, lmax + 1):
        for m in range(l + 1):
            ivec = plmidx(l, m)
            if m == 0:
                plm[l, m] = rc.al[l] * t * plm[l-1, 0] - rc.bl[l] * (u / 2) * plm[l-1, 1]
            elif m == lmax:
                plm[l, m] = u * rc.elm[ivec] * plm[l-1, m-1]
            else:
                plm[l, m] = rc.clm[ivec] * t * plm[l-1, m] - u * (rc.dlm[ivec] * plm[l-1, m+1] - rc.elm[ivec] * plm[l-1, m-1])

    return np.asarray(plm)


cpdef cnp.ndarray[double, ndim=3] fnALFs(double[:] rad_colat, int lmax):
    cdef:
        RecursiveCoef rc = RecursiveCoef(lmax)
        int nlat = rad_colat.shape[0]
        double[:,:,:] pilm = np.zeros((nlat, lmax + 1, lmax + 1))
        double t, u
        Py_ssize_t l, m
        int ivec
    
    rc.compute()
    for i in range(nlat):
        t = cos(rad_colat[i])
        u = sin(rad_colat[i])
        pilm[i, 0, 0] = 1.0
        pilm[i, 1, 0] = sqrt(3.0) * t
        pilm[i, 1, 1] = sqrt(3.0) * u
        for l in range(2, lmax + 1):
            for m in range(l + 1):
                ivec = plmidx(l, m)
                if m == 0:
                    pilm[i, l, m] = rc.al[l] * t * pilm[i, l-1, 0] - rc.bl[l] * (u / 2) * pilm[i, l-1, 1]
                elif m == lmax:
                    pilm[i, l, m] = u * rc.elm[ivec] * pilm[i, l-1, m-1]
                else:
                    pilm[i, l, m] = rc.clm[ivec]* t * pilm[i, l-1, m] - u * (rc.dlm[ivec] * pilm[i, l-1, m+1] - rc.elm[ivec] * pilm[i, l-1, m-1])
    
    return np.asarray(pilm)