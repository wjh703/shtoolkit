from functools import partial

import numpy as np

from .shtrans import grid2cilm
from .shtype import SpharmUnit, MassConserveMode, LoadLoveNumDict
from .shspecial import sea_level_equation, uniform_distributed

__all__ = ["SphereGrid"]


class SphereGrid:
    def __init__(
        self,
        data: np.ndarray,
        epochs: np.ndarray,
        unit: SpharmUnit,
        errors: np.ndarray | None = None,
        error_kind: str | None = None,
        name: str | None = None,
    ) -> None:
        if data.ndim == 2:
            data_3d = data[np.newaxis]
        else:
            data_3d = data
        if data_3d.shape[0] != epochs.shape[0]:
            msg = "The number of 'data' is unequal to that of 'epochs'"
            raise ValueError(msg)
        if errors is not None:
            if data.shape != errors.shape:
                msg = f"The shape of 'data' {data.shape}, is unequal to that of 'errors' {errors.shape}"
                raise ValueError(msg)

        max_resol = data.shape[-2] // 2 - 1

        self.data = data
        self.epochs = epochs
        self.unit: SpharmUnit = unit
        self.max_resol = max_resol
        self.errors = errors
        self.error_kind = error_kind
        self.name = name

    def expand(self, lmax_calc: int = -1):
        from .shcoeff import SpharmCoeff

        coeffs = np.array([grid2cilm(grid, lmax_calc) for grid in self.data])
        return SpharmCoeff(coeffs, self.epochs.copy(), self.unit)

    def conserve(
        self,
        oceanmask: np.ndarray,
        mode: MassConserveMode = "sal",
        lln: LoadLoveNumDict | None = None,
        lmax: int | None = None,
    ):
        if lmax is None:
            lmax = self.max_resol

        if mode == "eustatic":
            conserve_func = uniform_distributed
        elif mode == "sal" and lln is not None:
            conserve_func = partial(sea_level_equation, lln=lln, lmax=lmax, unit=self.unit, rot=False)  # type: ignore
        elif mode == "sal_rot" and lln is not None:
            conserve_func = partial(sea_level_equation, lln=lln, lmax=lmax, unit=self.unit, rot=True)  # type: ignore
        else:
            msg = f"'mode': {mode} needs a specific 'lln'"
            raise ValueError(msg)

        data = self.data
        if data.ndim == 2:
            if mode == "eustatic":
                data_conserve = conserve_func(data, oceanmask)
                s = "convsered from eustatic sea-level\n"
            else:
                data_conserve = conserve_func(data, oceanmask)[-1]
                s = "convsered from sea-level fingerprint\n"
            name = self.name + s if self.name is not None else s
        else:
            if mode == "eustatic":
                data_conserve = np.array([conserve_func(i, oceanmask) for i in data])
                s = "convsered from eustatic sea-level\n"
            else:
                data_conserve = np.array([conserve_func(i, oceanmask)[-1] for i in data])
                s = "convsered from sea-level fingerprint\n"
            name = self.name + s if self.name is not None else s
        return SphereGrid(data_conserve, self.epochs.copy(), self.unit, name=name)  # type: ignore
