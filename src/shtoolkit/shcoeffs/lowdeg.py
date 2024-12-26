import copy

import numpy as np
import numpy.typing as npt

from ..shread import read_technical_note_c20_c30, read_technical_note_deg1
from ..shtype import SpharmUnit
from ..shunit import SH_CONST
from .replacecoeff import ReplaceableCoeff

constant = copy.deepcopy(SH_CONST)
k2e = constant["k2e"]
kan = constant["kan"]
m_e = constant["m_e"]
moi_cm = constant["moi_cm"]
cmina = constant["cmina"]
a = constant["a"]


def c20tolod(c20: npt.ArrayLike):
    c20 = np.array(c20)
    factor = -(1 + k2e + kan) / (20**0.5 * m_e * a**2) * (3 * moi_cm / 0.997) / (1 + 1.125 * (k2e + kan))
    # factor = -2/3*M_e*a**2/moi_Cm
    lod = c20 / factor * 86400 * 1000
    return lod


def lodtoc20(lod: npt.ArrayLike):
    lod = np.array(lod)
    factor = -(1 + k2e + kan) / (20**0.5 * m_e * a**2) * (3 * moi_cm / 0.997) / (1 + 1.125 * (k2e + kan))
    # factor = -2/3*M_e*a**2/moi_Cm
    c20 = lod * factor / 86400 / 1000
    return c20


def cs21topm(cs21: npt.ArrayLike):
    cs21 = np.array(cs21)
    factor = -(1 + k2e) * (3 / 5) ** 0.5 * cmina / (1.098 * m_e * a**2)
    pm = cs21 / factor * 206265 * 1000
    return pm


def pmtocs21(pm: npt.ArrayLike):
    pm = np.array(pm)
    factor = -(1 + k2e) * (3 / 5) ** 0.5 * cmina / (1.098 * m_e * a**2)
    cs21 = pm * factor / 206265 / 1000
    return cs21


class Deg1(ReplaceableCoeff):
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        epochs: npt.ArrayLike,
        unit: SpharmUnit,
        errors: npt.ArrayLike | None = None,
        info: dict | None = None,
    ) -> None:
        if unit != "stokes":
            raise ValueError("Ensure C20 unit is stokes")
        indices = tuple(zip((0, 1, 0), (0, 1, 1), (1, 1, 1), strict=True))
        super().__init__(indice=indices, coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)

    @classmethod
    def from_technical_note_deg1(cls, filepath):
        epochs, deg1, deg1_sigma = read_technical_note_deg1(filepath)
        info = {"coeff": "DEG1", "version": "GRACE-OBP"}
        return cls(deg1, epochs, "stokes", deg1_sigma, info)


class C20(ReplaceableCoeff):
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        epochs: npt.ArrayLike,
        unit: SpharmUnit,
        errors: npt.ArrayLike | None = None,
        info: dict | None = None,
    ) -> None:
        if unit != "stokes":
            raise ValueError("Ensure C20 unit is stokes")
        super().__init__(indice=(0, 2, 0), coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)

    @classmethod
    def from_technical_note_c20(cls, filepath):
        epochs, c20, c20_sigma, _, _, center = read_technical_note_c20_c30(filepath)
        info = {"coeff": "C20", "version": center}
        return cls(c20, epochs, "stokes", c20_sigma, info)

    @classmethod
    def from_lod(cls, epochs: npt.ArrayLike, lod: npt.ArrayLike):
        coeff = lodtoc20(lod)
        return cls(coeff, epochs, "stokes")

    def to_lod(self):
        lod = c20tolod(self.coeffs)
        return lod

    def copy(self, **kwargs):
        copy_dict = copy.deepcopy(self.__dict__)
        copy_dict.pop("indice")
        if kwargs:
            for k, val in kwargs.items():
                copy_dict[k] = val
        return C20(**copy_dict)


class C30(ReplaceableCoeff):
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        epochs: npt.ArrayLike,
        unit: SpharmUnit,
        errors: npt.ArrayLike | None = None,
        info: dict | None = None,
    ) -> None:
        if unit != "stokes":
            raise ValueError("Ensure C30 unit is stokes")
        super().__init__(indice=(0, 3, 0), coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)

    @classmethod
    def from_technical_note_c30(cls, filepath):
        epochs, _, _, c30, c30_sigma, center = read_technical_note_c20_c30(filepath)
        info = {"coeff": "C30", "version": center}
        return cls(c30, epochs, "stokes", c30_sigma, info)


class C21(ReplaceableCoeff):
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        epochs: npt.ArrayLike,
        unit: SpharmUnit,
        errors: npt.ArrayLike | None = None,
        info: dict | None = None,
    ) -> None:
        if unit != "stokes":
            raise ValueError("Ensure C21 unit is stokes")
        super().__init__(indice=(0, 2, 1), coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)

    @classmethod
    def from_excitation(cls, epochs: npt.ArrayLike, excitation: npt.ArrayLike):
        coeff = pmtocs21(excitation)
        return cls(coeff, epochs, "stokes")

    def to_excitation(self):
        lod = cs21topm(self.coeffs)
        return lod


class S21(ReplaceableCoeff):
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        epochs: npt.ArrayLike,
        unit: SpharmUnit,
        errors: npt.ArrayLike | None = None,
        info: dict | None = None,
    ) -> None:
        if unit != "stokes":
            raise ValueError("Ensure S21 unit is stokes")
        super().__init__(indice=(1, 2, 1), coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)

    @classmethod
    def from_excitation(cls, epochs: npt.ArrayLike, excitation: npt.ArrayLike):
        coeff = pmtocs21(excitation)
        return cls(coeff, epochs, "stokes")

    def to_excitation(self):
        lod = cs21topm(self.coeffs)
        return lod
