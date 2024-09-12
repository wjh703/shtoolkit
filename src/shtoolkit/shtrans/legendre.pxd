# cython: language_level=3
cimport numpy as cnp

cnp.import_array()

cpdef cnp.ndarray[double, ndim=3] fnALFs(double[:] rad_colat, int lmax)
cpdef cnp.ndarray[double, ndim=2] fnALF(double rad_colat, int lmax)