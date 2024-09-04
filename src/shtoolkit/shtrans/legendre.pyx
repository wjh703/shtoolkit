# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

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
    """
    Reference:
        Xing, Z., Li, S., Tian, M. et al. Numerical experiments on column-wise recurrence formula to compute fully normalized
            associated Legendre functions of ultra-high degree and order. J Geod 94, 2 (2020). https://doi.org/10.1007/s00190-019-01331-0
    """
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