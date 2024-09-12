# cython: language_level=3
cimport numpy as cnp

cnp.import_array()

cpdef cnp.ndarray[double, ndim=1] cilm2vector(double[:,:,:] coeffs)
cpdef cnp.ndarray[double, ndim=3] vector2cilm(double[:] vector)