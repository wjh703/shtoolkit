import copy
from typing import Sequence

import numpy as np

from ..shload import (
    read_technical_note_c20_c30,
    read_technical_note_deg1,
)
from ..shtype import SpharmUnit
from ._harmonic import Harmonic
from .spharmcoeff import SpharmCoeff


class ReplaceCoeff(Harmonic):
    def __init__(
        self,
        indice: tuple[int, int, int] | Sequence[tuple[int, int, int]],
        coeffs: Sequence | np.ndarray,
        epochs: Sequence | np.ndarray,
        unit: SpharmUnit,
        errors: Sequence | np.ndarray | None = None,
        info: dict | None = None,
    ) -> None:
        super().__init__(coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)
        self.indice = indice

    def apply_to(self, sphcoef: SpharmCoeff):
        coeffs = np.copy(sphcoef.coeffs)
        errors = copy.deepcopy(sphcoef.errors)

        for ori_idx, t in enumerate(sphcoef.epochs):
            if self.indice == (0, 3, 0) and t < 2018:
                continue
            residual = np.abs(self.epochs - t)
            if np.nanmin(residual) > 0.05:
                msg = f"The epoch '{t:.4f}' cannot be found in the replaced epoch array."
                raise ValueError(msg)
            rp_idx = np.nanargmin(residual)
            coeffs[ori_idx, *self.indice] = self.coeffs[rp_idx]
            if self.errors is not None and errors is not None:
                errors[ori_idx, *self.indice] = self.errors[rp_idx]

        sphcoef_info = sphcoef.info.copy()
        if self.info:
            print(f"{self.info['coeff']} was replaced by {self.info['version']}.")
            sphcoef_info[self.info["coeff"]] = self.info["version"]

        sphcoef_new = sphcoef.copy(coeffs=coeffs, errors=errors, info=sphcoef_info)
        return sphcoef_new

    @classmethod
    def from_technical_note_c20(cls, filepath):
        indice = (0, 2, 0)
        epochs, c20, c20_sigma, _, _, center = read_technical_note_c20_c30(filepath)
        info = {"coeff": "C20", "version": center}
        return cls(indice, c20, epochs, "stokes", c20_sigma, info)

    @classmethod
    def from_technical_note_c30(cls, filepath):
        indice = (0, 3, 0)
        epochs, _, _, c30, c30_sigma, center = read_technical_note_c20_c30(filepath)
        info = {"coeff": "C30", "version": center}
        return cls(indice, c30, epochs, "stokes", c30_sigma, info)

    @classmethod
    def from_technical_note_deg1(cls, filepath):
        indice = tuple(zip((0, 1, 0), (0, 1, 1), (1, 1, 1), strict=False))
        epochs, deg1, deg1_sigma = read_technical_note_deg1(filepath)
        info = {"coeff": "DEG1", "version": "GRACE-OBP"}
        return cls(indice, deg1, epochs, "stokes", deg1_sigma, info)
