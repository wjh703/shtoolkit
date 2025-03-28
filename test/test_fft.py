import timeit

import numpy as np
import scipy as sp

from shtoolkit.shtrans.transform import _alloc_fftw_c2r, _alloc_fftw_r2c

if __name__ == "__main__":
    shape = (360, 720)
    grd = np.random.random(shape)
    a = np.fft.rfft(grd)
    b = sp.fft.rfft(grd)
    # a[:, 61:] = 0
    # b[:, 61:] = 0
    a1 = a.copy()
    a1[:, 1:] /= 2
    b1 = b.copy()
    b1[:, 1:] /= 2
    aa = np.fft.irfft(a1)
    bb = sp.fft.irfft(b1)

    if "rfft_object" not in globals().keys():
        global rfft_object
        rfft_object = _alloc_fftw_r2c(shape, (shape[0], shape[1] // 2 + 1))
    if "irfft_object" not in globals().keys():
        global irfft_object
        irfft_object = _alloc_fftw_c2r((shape[0], shape[1] // 2 + 1), shape)

    t1 = sp.fft.rfft(grd)
    t2 = rfft_object(grd)
    print(np.allclose(t1, t2))
    callable1 = lambda: np.fft.rfft(grd)
    callable2 = lambda: sp.fft.rfft(grd)
    callable3 = lambda: rfft_object(grd)
    print(timeit.timeit(callable1, number=1000))
    print(timeit.timeit(callable2, number=1000))
    print(timeit.timeit(callable3, number=1000))
    t3 = sp.fft.irfft(a1)
    t4 = irfft_object(a1)
    print(np.allclose(t1, t2))
    callable4 = lambda: np.fft.irfft(a1)
    callable5 = lambda: sp.fft.irfft(a1)
    callable6 = lambda: irfft_object(a1)
    print(timeit.timeit(callable4, number=1000))
    print(timeit.timeit(callable5, number=1000))
    print(timeit.timeit(callable6, number=1000))
