from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from ..shtrans import cilm2vector, vector2cilm


def read_slr_5x5(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
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

    vector_df = pd.DataFrame(np.hstack((np.asarray(epochs)[np.newaxis], vector))).rolling(4).mean().dropna().to_numpy()
    epochs_28d = vector_df[:, 0]
    vector_28d = vector_df[:, 1:]
    cilms_28d = np.array([vector2cilm(v) for v in vector_28d])

    return epochs_28d, cilms_28d
