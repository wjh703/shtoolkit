from typing import Literal, TypedDict

import numpy as np


class LoadLoveNumDict(TypedDict):
    h_el: np.ndarray
    l_el: np.ndarray
    k_el: np.ndarray


SpharmUnit = Literal[
    "mmewh",
    "mewh",
    "kmewh",
    "mmgeo",
    "mgeo",
    "kmgeo",
    "mmupl",
    "mupl",
    "kmupl",
    "kgm2mass",
    "stokes",
]
GIAModel = Literal["ICE6G-D", "ICE6G-C", "C18"]
SHSmoothKind = Literal["gauss", "fan"]
