import numpy as np
import pywt


def wl_decompose(signal: np.ndarray, wname: str, level: int = 4):
    n = signal.size
    coeffs = pywt.wavedec(signal, wname, level=level)
    ca = coeffs[0]
    cd = reversed(coeffs[1:])
    rec_a = pywt.waverec([ca] + [None] * level, wname)[:n]  # 近似系数重构
    rec_d = [pywt.waverec([None, coeff] + [None] * i, wname)[:n] for i, coeff in enumerate(cd)]  # 细节系数重构
    rec = np.r_[rec_d, [rec_a]]
    # rec = np.asarray(rec_d)
    # rec[-1] += rec_a
    rec = np.flip(rec, axis=0)
    return rec
