import numpy as np


__all__ = ["gauss_smooth", "fan_smooth"]


def gauss_smooth(lmax: int, radius: int = 300) -> np.ndarray:
	ww = np.zeros(lmax + 1, dtype=np.float64)
	bb = np.log(2) / (1 - np.cos(radius / 6371))
	ww[0] = 1
	ww[1] = (1 + np.exp(-2 * bb)) / (1 - np.exp(-2 * bb)) - 1 / bb
	for i in range(2, len(ww)):
		ww[i] = -(2 * i - 1) / bb * ww[i - 1] + ww[i - 2]
	gauss_ww = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
	for i in range(lmax + 1):
		j = np.arange(i + 1)
		gauss_ww[i, j] = ww[i]
	return np.asarray([gauss_ww, gauss_ww])


def fan_smooth(lmax: int, radius: int = 300) -> np.ndarray:
	ww = np.zeros(lmax + 1, dtype=np.float64)
	bb = np.log(2) / (1 - np.cos(radius / 6371))
	ww[0] = 1
	ww[1] = (1 + np.exp(-2 * bb)) / (1 - np.exp(-2 * bb)) - 1 / bb
	for i in range(2, len(ww)):
		ww[i] = -(2 * i - 1) / bb * ww[i - 1] + ww[i - 2]
	fan_ww = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
	for i in range(lmax + 1):
		j = np.arange(i + 1)
		fan_ww[i, j] = ww[i] * ww[j]
	return np.asarray([fan_ww, fan_ww])
