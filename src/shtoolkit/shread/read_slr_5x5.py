from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from ..shtime import year_month_to_decimal_year
from ..shtrans import cilm2vector, vector2cilm


def read_slr_5x5(filepath: str | Path):
    def read_gsfc_5x5(ff: TextIO):
        cilm = np.zeros((2, 7, 7))
        for _ in range(19):
            line = next(ff).strip().split()
            l, m, c, s = int(line[0]), int(line[1]), float(line[2]), float(line[3])
            cilm[0, l, m] = c
            cilm[1, l, m] = s
        return cilm

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("Product"):
                break
        cilms = []
        epochs = []
        for line in f:
            ls = line.strip().split()
            epochs.append(float(ls[1]))
            cilms.append(read_gsfc_5x5(f))

    vector = np.array([cilm2vector(c) for c in np.asarray(cilms)])
    vector_df = (
        pd.DataFrame(np.hstack((np.asarray(epochs)[:, np.newaxis], vector))).rolling(4).mean().dropna().to_numpy()
    )
    epochs_28d = vector_df[:, 0]
    vector_28d = vector_df[:, 1:]
    cilms_28d = np.array([vector2cilm(v) for v in vector_28d])
    from ..shcoeffs import SpharmCoeff

    return SpharmCoeff(cilms_28d, epochs_28d, "stokes")


def read_csr_5x5(filepath: str | Path):
    with open(filepath, "r") as f:
        for line in f:
            if "2I5,2D20.12,2D13.5" in line:
                next(f)
                next(f)
                next(f)
                break

        cilm_mean = np.zeros((2, 7, 7))
        for _ in range(19):
            line = next(f).strip().replace("D", "E").split()
            l, m, c, s = (
                int(line[0]),
                int(line[1]),
                float(line[2]),
                float(line[3]),
            )
            cilm_mean[0, l, m] = c
            cilm_mean[1, l, m] = s

        for line in f:
            if "end of header" in line:
                break

        cilms = []
        cilms_errors = []
        epochs = []

        for line in f:
            ls = line.strip().split()
            year, month = int(ls[3]), int(ls[4])
            decimal_year = year_month_to_decimal_year(f"{year:04d}{month:02d}")
            epochs.append(decimal_year)

            cilm = np.zeros((2, 7, 7))
            cilm_errors = np.zeros((2, 7, 7))
            for _ in range(19):
                line = next(f).strip().split()
                l, m, c, s, c_e, s_e = (
                    int(line[0]),
                    int(line[1]),
                    float(line[2]),
                    float(line[3]),
                    float(line[6]),
                    float(line[7]),
                )
                cilm[0, l, m] = c * 1e-10
                cilm[1, l, m] = s * 1e-10
                cilm_errors[0, l, m] = c_e * 1e-10
                cilm_errors[1, l, m] = s_e * 1e-10
            cilms.append(cilm + cilm_mean)
            cilms_errors.append(cilm_errors)
    from ..shcoeffs import SpharmCoeff

    return SpharmCoeff(cilms, epochs, "stokes", cilms_errors)
