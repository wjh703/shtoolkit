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


class SpharmCoeff(Harmonic):
    def __init__(
        self,
        coeffs: Sequence | np.ndarray,
        epochs: Sequence | np.ndarray,
        unit: SpharmUnit,
        errors: Sequence | np.ndarray | None = None,
        info: dict | None = None,
    ) -> None:
        super().__init__(coeffs=coeffs, epochs=epochs, unit=unit, errors=errors, info=info)
        self.lmax = self.coeffs.shape[-2] - 1

    @classmethod
    def from_files(
        cls,
        folder: str,
        lmax: int,
        is_icgem: bool = True,
    ):

        files = [file for file in Path(folder).iterdir()]

        if is_icgem:
            data = [read_icgem(file, lmax) for file in files]
        else:
            data = [read_non_icgem(file, lmax) for file in files]
        epochs, coeffs, errors = map(np.array, zip(*data, strict=False))

        center = re.findall(r"UTCSR|GFZOP|JPLEM|COSTG|GRGS|AIUB|ITSG|HUST|Tongji", files[0].stem)
        info = dict(GSM=center[0]) if center else None
        return cls(coeffs, epochs, "stokes", errors, info=info).sort()

    def replace(
        self,
        replow: dict | Sequence[dict],
        rpcoef=None,
    ):
        from .replacecoeff import ReplaceCoeff

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
                    sphcoef = repcoef.apply_to(sphcoef)  # type: ignore
        else:
            sphcoef = rpcoef.apply_to(self)
        return sphcoef

    def corr_gia(
        self,
        modelname: GIAModel,
        filepath: str | Path,
        mode: Literal["add", "subtract"] = "subtract",
    ):
        if mode not in ["add", "subtract"]:
            msg = f"Invalid value of 'mode', (expected 'subtract' or 'add', got '{mode}'"
            raise ValueError(msg)

        gia_trend = read_gia_model(filepath, self.lmax, modelname)
        gia_coeffs = np.array([epoch * gia_trend for epoch in self.epochs])
        gia_coeffs -= gia_coeffs.mean(axis=0)

        info = self.info.copy()
        if mode == "subtract":
            coeffs = self.coeffs - gia_coeffs
            info["GIA"] = modelname
        elif mode == "add":
            coeffs = self.coeffs + gia_coeffs

        print(f"GIA was {mode}ed by {modelname}.")

        sphcoef_new = self.copy(coeffs=coeffs, info=info)
        return sphcoef_new

    def sort(self):
        coeffs_sorted = self.coeffs[np.argsort(self.epochs)]
        if self.errors is not None:
            errors_sorted = self.errors[np.argsort(self.epochs)]
        else:
            epochs_sorted = None

        epochs_sorted = np.sort(self.epochs)
        sphcoef_new = self.copy(coeffs=coeffs_sorted, errors=errors_sorted, epochs=epochs_sorted)
        return sphcoef_new

    def remove_mean_field(self):
        coeffs = self.coeffs - self.coeffs.mean(axis=0)
        sphcoef_new = self.copy(coeffs=coeffs)
        return sphcoef_new

    def smooth(self, kind: SHSmoothKind = "gauss", radius: int = 300):
        if kind == "gauss":
            weight = gauss_smooth(self.lmax, radius)
        elif kind == "fan":
            weight = fan_smooth(self.lmax, radius)
        coeffs = self.coeffs * weight

        sphcoef_new = self.copy(coeffs=coeffs)
        return sphcoef_new

    def expand(self, resol: int, lmax_calc: int = -1):
        from ..shgrid import SphereGrid

        data = np.asarray([cilm2grid(cilm, resol, lmax_calc) for cilm in self.coeffs])
        return SphereGrid(data, self.epochs.copy(), self.unit)

    def convert(self, new_unit: SpharmUnit, lln: LoadLoveNumDict | None = None):
        coeffs = unitconvert(self.coeffs, self.unit, new_unit, lln)
        sphcoef_new = self.copy(coeffs=coeffs, unit=new_unit)
        return sphcoef_new
