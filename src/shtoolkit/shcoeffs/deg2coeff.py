import re
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from ..shfilter import fan_smooth, gauss_smooth
from ..shload import (
    read_gia_model,
    read_icgem,
    read_non_icgem,
)
from ..shtrans import cilm2grid
from ..shtype import GIAModel, LoadLoveNumDict, SHSmoothKind, SpharmUnit
from ..shunit import unitconvert
from ._harmonic import Harmonic


class Deg2(Harmonic):
    def __init__(
        self,
        coeffs: Sequence | np.ndarray,
        epochs: Sequence | np.ndarray,
        unit: SpharmUnit,
        errors: Sequence | np.ndarray | None = None,
        info: dict | None = None,
    ) -> None:
        super().__init__(coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)

