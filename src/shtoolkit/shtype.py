from typing import Literal, TypedDict

import numpy as np


__all__ = [
    "LoadLoveNumDict", "LeakCorrMethod", 
    "SpharmUnit", "GIAModel", 
    "SHSmoothKind", "MassConserveMode"
]

class LoadLoveNumDict(TypedDict):
    h_el: np.ndarray
    l_el: np.ndarray
    k_el: np.ndarray


class LeakCorrMethod(TypedDict):
    method: Literal["buf", "buf_gs", "buf_fs", "FM_gs", "FM_fs"]
    radius: int | None


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
MassConserveMode = Literal["eustatic", "sal", "sal_rot"]
