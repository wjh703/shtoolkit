import copy
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from ..shload import read_gia_model
from ..shtype import GIAModel, SpharmUnit
from ..shunit import SH_CONST
from ._harmonic import Harmonic

constant = copy.deepcopy(SH_CONST)
k2e = constant["k2e"]
kan = constant["kan"]
m_e = constant["m_e"]
moi_cm = constant["moi_cm"]
cmina = constant["cmina"]
a = constant["a"]


def c20tolod(c20):
    factor = -(1 + k2e + kan) / (20**0.5 * m_e * a**2) * (3 * moi_cm / 0.997) / (1 + 1.125 * (k2e + kan))
    # factor = -2/3*M_e*a**2/moi_Cm
    lod = c20 / factor * 86400 * 1000
    return lod


def lodtoc20(lod):
    factor = -(1 + k2e + kan) / (20**0.5 * m_e * a**2) * (3 * moi_cm / 0.997) / (1 + 1.125 * (k2e + kan))
    # factor = -2/3*M_e*a**2/moi_Cm
    c20 = lod * factor * 86400 * 1000
    return c20


def cs21topm(cs21):
    factor = -(1 + k2e) * (3 / 5) ** 0.5 * cmina / (1.098 * m_e * a**2)
    pm = cs21 / factor * 206265 * 1000
    return pm


def pmtocs21(pm):
    factor = -(1 + k2e) * (3 / 5) ** 0.5 * cmina / (1.098 * m_e * a**2)
    cs21 = pm * factor / 206265 / 1000
    return cs21


class Deg2(Harmonic):
    def __init__(
        self,
        indice: tuple[int, int, int],
        coeffs: Sequence | np.ndarray,
        epochs: Sequence | np.ndarray,
        unit: SpharmUnit,
        errors: Sequence | np.ndarray | None = None,
        info: dict | None = None,
    ) -> None:
        if unit != "stokes":
            raise ValueError("Confirm the unit of deg2 coeff is stokes")
        super().__init__(coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)
        self.indice = indice

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
                "The indice of coeff can not deal with gia pole tide, "
                + f"(got {self.indice}, expected (0, 2, 1) or (1, 2, 1))"
            )
            raise ValueError(msg)
        gia_pole_tide -= gia_pole_tide.mean()

        if mode == "add":
            coeffs = self.coeffs + gia_pole_tide
        else:
            coeffs = self.coeffs - gia_pole_tide

        coeffs -= coeffs.mean()
        return self.copy(coeffs=coeffs)

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

    @classmethod
    def from_eops(cls, epochs, eops, kind: Literal["x_excitation", "y_excitation", "lod"]):
        if kind == "x_excitation":
            coeff = pmtocs21(eops)
            indice = (0, 2, 1)
        elif kind == "y_excitation":
            coeff = pmtocs21(eops)
            indice = (1, 2, 1)
        elif kind == "lod":
            coeff = lodtoc20(eops)
            indice = (0, 2, 0)
        return cls(indice, coeff, epochs, "stokes")

    def to_eops(self):
        if self.indice in [(0, 2, 1), (1, 2, 1)]:
            eops = cs21topm(self.coeffs.ravel())
        elif self.indice == (0, 2, 1):
            eops = c20tolod(self.coeffs.ravel())
        return eops

    def copy(self, **kwargs):
        copy_dict = copy.deepcopy(self.__dict__)
        if kwargs:
            for k, val in kwargs.items():
                copy_dict[k] = val
        return Deg2(**copy_dict)
