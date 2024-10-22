import copy
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from ..shread import read_gia_model
from ..shtype import GIAModel, SpharmUnit
from .harmonic import Harmonic
from .spharmcoeff import SpharmCoeff


class ReplaceableCoeff(Harmonic):
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

    def corr_gia(
        self,
        modelname: GIAModel,
        filepath: str | Path,
        mode: Literal["add", "subtract"] = "subtract",
    ):
        if mode not in ["add", "subtract"]:
            msg = f"Invalid value of 'mode', (expected 'subtract' or 'add', got '{mode}'"
            raise ValueError(msg)

        gia_trend = read_gia_model(filepath, 2, modelname)[*self.indice]
        gia_coeffs = np.array([epoch * gia_trend for epoch in self.epochs])[:, np.newaxis]
        gia_coeffs -= gia_coeffs.mean(axis=0)

        if mode == "subtract":
            coeffs = self.coeffs - gia_coeffs
        elif mode == "add":
            coeffs = self.coeffs + gia_coeffs

        return self.copy(coeffs=coeffs)

    def corr_gia_pole_tide(self, mode: Literal["add", "subtract"] = "subtract", reference_time: float = 2000.0):
        # m1_gia = 5.5e-2 + 1.677e-3 * (self.dtime - reference_time)
        # m2_gia = -0.3205 - 3.46e-3 * (self.dtime - reference_time)

        m1_gia = 1.677e-3 * (self.epochs - reference_time)[:, np.newaxis]
        m2_gia = -3.46e-3 * (self.epochs - reference_time)[:, np.newaxis]

        if self.indice == (0, 2, 1):
            gia_pole_tide = -1.551e-9 * m1_gia - 0.012e-9 * m2_gia
        elif self.indice == (1, 2, 1):
            gia_pole_tide = 0.021e-9 * m1_gia - 1.505e-9 * m2_gia
        else:
            msg = (
                "Invalid indice which the coeff cannot deal with gia pole tide, "
                + f"(got {self.indice}, expected (0, 2, 1) or (1, 2, 1))"
            )
            raise ValueError(msg)

        gia_pole_tide -= gia_pole_tide.mean()

        if mode == "add":
            coeffs = self.coeffs + gia_pole_tide
        else:
            coeffs = self.coeffs - gia_pole_tide

        coeffs -= coeffs.mean()  # type: ignore
        return self.copy(coeffs=coeffs)

    def copy(self, **kwargs):
        copy_dict = copy.deepcopy(self.__dict__)
        if issubclass(self.__class__, ReplaceableCoeff):
            # if self.__class__ in ReplaceableCoeff.__subclasses__():
            copy_dict.pop("indice")
        if kwargs:
            for k, val in kwargs.items():
                copy_dict[k] = val
        return self.__class__(**copy_dict)
