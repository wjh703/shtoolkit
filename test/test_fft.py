import timeit

import numpy as np
import scipy as sp
import pyfftw

if __name__ == "__main__":

    shape = (360, 720)
    grd = np.random.random(shape)
    a = np.fft.rfft(grd)
    b = sp.fft.rfft(grd)
    a[:, 61:] = 0
    b[:, 61:] = 0
    a1 = a.copy()
    a1[:, 1:] /= 2
    b1 = b.copy()
    b1[:, 1:] /= 2
    aa = np.fft.irfft(a1)
    bb = sp.fft.irfft(b1)

    rfftin = pyfftw.empty_aligned(shape, dtype="float64")
    rfftout = pyfftw.empty_aligned((shape[0], shape[1] // 2 + 1), dtype="complex128")

    rfft_object = pyfftw.FFTW(rfftin, rfftout)
    c = rfft_object(grd).copy()
    d = rfft_object(np.random.random(shape)).copy()
    c[:, 61:] = 0
    print(np.allclose(a, c))

    irfftin = pyfftw.empty_aligned((shape[0], shape[1] // 2 + 1), dtype="complex128")
    irfftout = pyfftw.empty_aligned(shape, dtype="float64")
    irfft_object = pyfftw.FFTW(irfftin, irfftout, direction="FFTW_BACKWARD")
    irfftin[:] = a1
    cc = irfft_object()
    print(np.allclose(aa, cc))
    # breakpoint()
    callable1 = lambda: np.fft.rfft(grd)
    callable2 = lambda: sp.fft.rfft(grd)
    callable3 = lambda: rfft_object()
    print(timeit.timeit(callable1, number=1000))
    print(timeit.timeit(callable2, number=1000))
    print(timeit.timeit(callable3, number=1000))

    callable4 = lambda: np.fft.irfft(a1)
    callable5 = lambda: sp.fft.irfft(b1)
    callable6 = lambda: irfft_object()
    print(timeit.timeit(callable4, number=1000))
    print(timeit.timeit(callable5, number=1000))
    print(timeit.timeit(callable6, number=1000))

    callable7 = lambda: sp.fft.irfft(b1)
    callable8 = lambda: sp.fft.ifft(b).real
    print(timeit.timeit(callable7, number=1000))
    print(timeit.timeit(callable8, number=1000))
