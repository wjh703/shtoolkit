import copy
import re
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from .shfilter import fan_smooth, gauss_smooth
from .shload import (
    read_gia_model,
    read_icgem,
    read_non_icgem,
    read_technical_note_c20_c30,
    read_technical_note_deg1,
)
from .shtrans import cilm2grid
from .shtype import GIAModel, LoadLoveNumDict, RepFileDict, RepInsDict, SHSmoothKind, SpharmUnit
from .shunit import convert


class SpharmCoeff:
    def __init__(
        self,
        coeffs: np.ndarray,
        epochs: np.ndarray | float,
        unit: SpharmUnit,
        errors: np.ndarray | None = None,
        error_kind: str | None = None,
        name: str | None = None,
    ) -> None:
        if errors is not None:
            if coeffs.shape != errors.shape:
                msg = f"The shape of 'coeffs' {coeffs.shape}, is unequal to that of 'errors' {errors.shape}"
                raise ValueError(msg)

        if coeffs.ndim == 4:
            self.coeffs = coeffs.copy()
            self.errors = errors.copy() if errors is not None else None
        elif coeffs.ndim == 3:
            self.coeffs = coeffs.copy()[np.newaxis]
            self.errors = errors.copy()[np.newaxis] if errors is not None else None
        else:
            msg = f"Invalid ndim of 'coeffs', (expected 3 or 4, got {coeffs.ndim})"
            raise ValueError(msg)

        epochs = epochs.copy() if isinstance(epochs, np.ndarray) else np.array([epochs])
        if self.coeffs.shape[0] != epochs.shape[0]:
            msg = (
                f"The number of 'coeffs' {self.coeffs.shape[0]} ," + "is unequal to that of 'epochs' {epochs.shape[0]}"
            )
            raise ValueError(msg)

        self.lmax = coeffs.shape[-2] - 1
        self.epochs = epochs
        self.unit: SpharmUnit = unit
        self.error_kind = error_kind
        self.name = name

    @classmethod
    def from_files(
        cls,
        folder: str | Path,
        lmax: int,
        is_icgem: bool = True,
    ) -> "SpharmCoeff":
        if isinstance(folder, str):
            folder = Path(folder)

        files = [file for file in folder.iterdir()]

        if is_icgem:
            data = [read_icgem(file, lmax) for file in files]
        else:
            data = [read_non_icgem(file, lmax) for file in files]
        epochs, coeffs, errors = map(np.array, zip(*data, strict=False))

        center = re.findall(r"UTCSR|GFZOP|JPLEM|COSTG|GRGS|AIUB|ITSG|HUST|Tongji", files[0].stem)
        name = f"GSM: {center[0]}\n" if center else None
        return cls(coeffs, epochs, "stokes", errors, name=name).sort()

    def replace(
        self,
        replow: RepFileDict | Sequence[RepFileDict],
        rpcoef: Optional["ReplaceCoeff"] = None,
    ) -> "SpharmCoeff":
        if self.unit != "stokes":
            msg = f"Invalid value of 'unit' attribute, (expected 'stokes', got '{self.unit}')."
            raise ValueError(msg)

        if rpcoef is None:
            lowdeg_dict = {
                "C20": ReplaceCoeff.from_technical_note_c20,
                "C30": ReplaceCoeff.from_technical_note_c30,
                "DEG1": ReplaceCoeff.from_technical_note_deg1,
            }
            if isinstance(replow, dict):
                repcoef = lowdeg_dict[replow["rep"]](replow["file"])
                sphcoef = repcoef.apply_to(self)
            else:
                sphcoef = self
                for i in range(len(replow)):
                    repcoef = lowdeg_dict[replow[i]["rep"]](replow[i]["file"])
                    sphcoef = repcoef.apply_to(sphcoef)
        else:
            sphcoef = rpcoef.apply_to(self)
        return sphcoef

    def corr_gia(
        self,
        modelname: GIAModel,
        filepath: str | Path,
        mode: Literal["add", "subtract"] = "subtract",
    ) -> "SpharmCoeff":
        if mode not in ["add", "subtract"]:
            msg = f"Invalid value of 'mode', (expected 'subtract' or 'add', got '{mode}'"
            raise ValueError(msg)

        gia_trend = read_gia_model(filepath, self.lmax, modelname)
        gia_coeffs = np.array([epoch * gia_trend for epoch in self.epochs])
        gia_coeffs -= gia_coeffs.mean(axis=0)

        name = self.name
        if mode == "subtract":
            coeffs = self.coeffs - gia_coeffs
            if isinstance(name, str):
                name += f"GIA: {modelname}\n"
        elif mode == "add":
            coeffs = self.coeffs + gia_coeffs

        print(f"GIA was {mode}ed by {modelname}.")

        sphcoef_new = self.copy(coeffs=coeffs, name=name)
        return sphcoef_new

    def sort(self) -> "SpharmCoeff":
        coeffs_sorted = self.coeffs[np.argsort(self.epochs)]
        if self.errors is not None:
            errors_sorted = self.errors[np.argsort(self.epochs)]
        else:
            epochs_sorted = None

        epochs_sorted = np.sort(self.epochs)
        sphcoef_new = self.copy(coeffs=coeffs_sorted, errors=errors_sorted, epochs=epochs_sorted)
        return sphcoef_new

    def remove_mean_field(self) -> "SpharmCoeff":
        coeffs = self.coeffs - self.coeffs.mean(axis=0)
        sphcoef_new = self.copy(coeffs=coeffs)
        return sphcoef_new

    def smooth(self, kind: SHSmoothKind = "gauss", radius: int = 300) -> "SpharmCoeff":
        if kind == "gauss":
            weight = gauss_smooth(self.lmax, radius)
        elif kind == "fan":
            weight = fan_smooth(self.lmax, radius)
        coeffs = self.coeffs * weight

        sphcoef_new = self.copy(coeffs=coeffs)
        return sphcoef_new

    def expand(self, resol: int, lmax_calc: int = -1):
        from .shgrid import SphereGrid

        data = np.array([cilm2grid(cilm, resol, lmax_calc) for cilm in self.coeffs])
        return SphereGrid(data, self.epochs.copy(), self.unit)

    def unitconvert(self, new_unit: SpharmUnit, lln: LoadLoveNumDict | None = None) -> "SpharmCoeff":
        coeffs = convert(self.coeffs, self.unit, new_unit, lln)
        sphcoef_new = self.copy(coeffs=coeffs, unit=new_unit)
        return sphcoef_new

    def copy(self, **kwargs) -> "SpharmCoeff":
        sphcoef_dict_copy = copy.deepcopy(self.__dict__)
        del sphcoef_dict_copy["lmax"]
        if kwargs:
            for k, val in kwargs.items():
                sphcoef_dict_copy[k] = val
        return SpharmCoeff(**sphcoef_dict_copy)

    def __getitem__(self, index) -> "SpharmCoeff":
        if self.errors is None:
            sphcoef_new = self.copy(coeffs=self.coeffs[index], epochs=self.epochs[index])
        else:
            sphcoef_new = self.copy(coeffs=self.coeffs[index], epochs=self.epochs[index], errors=self.errors[index])
        return sphcoef_new

    def __len__(self) -> int:
        return self.coeffs.shape[0]

    def __add__(self, other: "SpharmCoeff") -> "SpharmCoeff":
        if self.coeffs.shape == other.coeffs.shape and np.allclose(self.epochs, other.epochs, atol=0.5):
            coeffs = self.coeffs + other.coeffs
        else:
            msg = (
                f"The shape of 'coeffs' is unequal (got {self.coeffs.shape} and {other.coeffs.shape}), "
                + "or the 'epochs' is not close."
            )
            raise ValueError(msg)
        sphcoef_new = self.copy(coeffs=coeffs)
        return sphcoef_new

    def __sub__(self, other: "SpharmCoeff") -> "SpharmCoeff":
        if self.coeffs.shape == other.coeffs.shape and np.allclose(self.epochs, other.epochs, atol=0.5):
            coeffs = self.coeffs - other.coeffs
        else:
            msg = (
                f"The shape of 'coeffs' is unequal (got {self.coeffs.shape} and {other.coeffs.shape}), "
                + "or the 'epochs' is not close."
            )
            raise ValueError(msg)

        sphcoef_new = self.copy(coeffs=coeffs)
        return sphcoef_new


class ReplaceCoeff:
    def __init__(
        self,
        indice: Sequence[int] | Sequence[Sequence[int]],
        coeffs: np.ndarray,
        epochs: np.ndarray,
        unit: SpharmUnit,
        errors: np.ndarray | None = None,
        name: RepInsDict | None = None,
    ) -> None:
        self.indice = indice
        self.coeffs = coeffs
        self.epochs = epochs
        self.unit: SpharmUnit = unit
        self.errors = errors
        self.name = name

    def apply_to(self, sphcoef: SpharmCoeff) -> SpharmCoeff:
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

        if isinstance(self.name, dict):
            print(f"{self.name['rep']} was replaced by {self.name['institute']}.")
            sphname = (
                sphcoef.name + f"{self.name['rep']}: {self.name['institute']}\n" if sphcoef.name is not None else None
            )
        elif self.name is None:
            sphname = sphcoef.name
        else:
            msg = f"Invalid attribute of name <{self.name}>."
            raise AttributeError(msg)

        sphcoef_new = sphcoef.copy(coeffs=coeffs, errors=errors, name=sphname)
        return sphcoef_new

    @classmethod
    def from_technical_note_c20(cls, filepath):
        indice = (0, 2, 0)
        epochs, c20, c20_sigma, _, _, center = read_technical_note_c20_c30(filepath)
        info: RepInsDict = {"rep": "C20", "institute": center}
        return cls(indice, c20, epochs, "stokes", c20_sigma, info)

    @classmethod
    def from_technical_note_c30(cls, filepath):
        indice = (0, 3, 0)
        epochs, _, _, c30, c30_sigma, center = read_technical_note_c20_c30(filepath)
        info: RepInsDict = {"rep": "C30", "institute": center}
        return cls(indice, c30, epochs, "stokes", c30_sigma, info)

    @classmethod
    def from_technical_note_deg1(cls, filepath):
        indice = tuple(zip((0, 1, 0), (0, 1, 1), (1, 1, 1), strict=False))
        epochs, deg1, deg1_sigma = read_technical_note_deg1(filepath)
        info: RepInsDict = {"rep": "DEG1", "institute": "GRACE-OBP"}
        return cls(indice, deg1, epochs, "stokes", deg1_sigma, info)
