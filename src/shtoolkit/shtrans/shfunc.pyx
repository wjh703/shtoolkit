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

cpdef cnp.ndarray[double, ndim=1] cilm2vector(double[:,:,:] coeffs):
    cdef: 
        int lmax = coeffs.shape[1] - 1
        int cilm_num = (lmax + 1)**2
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
        double[:,:,:] coeffs = np.zeros((2, lmax + 1, lmax + 1))
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


def shreal2complex(double[:,:,:] cilm):
    cdef:
        int lmax = cilm.shape[1] - 1
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


def shcomplex2real(double[:,:,:] cilm_complex):
    cdef:
        int lmax = cilm_complex.shape[1] - 1
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
