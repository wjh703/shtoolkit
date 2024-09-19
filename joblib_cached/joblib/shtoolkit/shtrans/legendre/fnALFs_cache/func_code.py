# first line: 28
@mem.cache
def fnALFs_cache(cnp.ndarray[double, ndim=1] rad_colat, int lmax):
    return fnALFs(rad_colat, lmax)
