import copy
from typing import Sequence

import numpy as np

from ..shtype import SpharmUnit


class Harmonic:
    def __init__(
        self,
        coeffs: Sequence | np.ndarray,
        epochs: Sequence | np.ndarray,
        unit: SpharmUnit,
        errors: Sequence | np.ndarray | None = None,
        info: dict | None = None,
    ) -> None:
        """
        A base class working with time-varying spherical harmonic coefficients.
        """
        coeffs_ndarray = np.array(coeffs)
        epochs_ndarray = np.array(epochs)
        if errors is not None:
            errors_ndarray = np.array(errors)
            if coeffs_ndarray.shape != errors_ndarray.shape:
                msg = (
                    "Invalid shapes between 'coeffs' and 'errors' "
                    + f"(got {coeffs_ndarray.shape} and {errors_ndarray.shape}"
                )
                raise ValueError(msg)

        if coeffs_ndarray.ndim == 3:
            self.coeffs = coeffs_ndarray.copy()[np.newaxis]
            self.errors = errors_ndarray.copy()[np.newaxis] if errors is not None else None
        elif coeffs_ndarray.ndim == 1:
            self.coeffs = coeffs_ndarray.copy()[:, np.newaxis]
            self.errors = errors_ndarray.copy()[:, np.newaxis] if errors is not None else None
        elif coeffs_ndarray.ndim in [2, 4]:
            self.coeffs = coeffs_ndarray.copy()
            self.errors = errors_ndarray.copy() if errors is not None else None
        else:
            msg = f"Invalid ndim of 'coeffs', (expected 1, 2, 3 or 4, got {coeffs_ndarray.ndim})"
            raise ValueError(msg)

        if self.coeffs.shape[0] != epochs_ndarray.shape[0]:
            msg = (
                "Invalid length between 'coeffs' and 'epochs', "
                + f"(got{self.coeffs.shape[0]} and {epochs_ndarray.shape[0]})"
            )
            raise ValueError(msg)

        self.epochs = epochs_ndarray
        self.unit: SpharmUnit = unit
        self.info = info if info is not None else dict()

    def copy(self, **kwargs):
        copy_dict = copy.deepcopy(self.__dict__)
        if kwargs:
            for k, val in kwargs.items():
                copy_dict[k] = val
        return self.__class__(**copy_dict)

    def __getitem__(self, index):
        if self.errors is None:
            harmonic_slice = self.copy(coeffs=self.coeffs[index], epochs=self.epochs[index])
        else:
            harmonic_slice = self.copy(coeffs=self.coeffs[index], epochs=self.epochs[index], errors=self.errors[index])
        return harmonic_slice

    def __len__(self) -> int:
        return self.coeffs.shape[0]

    def __add__(self, other):
        if (
            self.coeffs.shape == other.coeffs.shape
            and np.allclose(self.epochs, other.epochs, atol=0.05)
            and self.__class__ == other.__class__
        ):
            coeffs = self.coeffs + other.coeffs
        else:
            msg = (
                f"The shape of 'coeffs' is unequal (got {self.coeffs.shape} and {other.coeffs.shape}), "
                + "or the 'epochs' is not close, or the two instances are not from same class."
            )
            raise ValueError(msg)
        harmonic_add = self.copy(coeffs=coeffs)
        return harmonic_add

    def __sub__(self, other):
        if (
            self.coeffs.shape == other.coeffs.shape
            and np.allclose(self.epochs, other.epochs, atol=0.05)
            and self.__class__ == other.__class__
        ):
            coeffs = self.coeffs - other.coeffs
        else:
            msg = (
                f"The shape of 'coeffs' is unequal (got {self.coeffs.shape} and {other.coeffs.shape}), "
                + "or the 'epochs' is not close."
            )
            raise ValueError(msg)

        harmonic_sub = self.copy(coeffs=coeffs)
        return harmonic_sub
