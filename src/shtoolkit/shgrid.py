from functools import partial

import numpy as np

from .shspecial import sea_level_equation, uniform_distributed
from .shtrans import grid2cilm
from .shtype import LoadLoveNumDict, MassConserveMode, SpharmUnit


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
        from .shcoeffs import SpharmCoeff

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
            msg = (
                f"Invalid value of 'mode' (expected 'eustatic', 'sal' or 'sal_rot', got '{mode}'), "
                + "or 'lln' (expected 'LoadLoveNumDict', got None)"
            )
            raise ValueError(msg)

        data = self.data
        if mode == "eustatic":
            data_conserve = (
                conserve_func(data, oceanmask)
                if data.ndim == 2
                else np.asarray([conserve_func(i, oceanmask) for i in data])
            )
            s = "convsered from eustatic sea-level\n"
        else:
            data_conserve = (
                conserve_func(data, oceanmask)[-1]
                if data.ndim == 2
                else np.asarray([conserve_func(i, oceanmask)[-1] for i in data])
            )
            s = "convsered from sea-level fingerprint\n"
        name = self.name + s if self.name is not None else s
        return SphereGrid(data_conserve, self.epochs.copy(), self.unit, name=name)  # type: ignore
