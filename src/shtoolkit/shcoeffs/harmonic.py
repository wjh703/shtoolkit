import copy

import numpy as np
import numpy.typing as npt

from ..shtype import SpharmUnit
from ..shtrans import cilm2vector, vector2cilm

class Harmonic:
    def __init__(
        self,
        coeffs: npt.ArrayLike,
        epochs: npt.ArrayLike,
        unit: SpharmUnit,
        errors: npt.ArrayLike | None = None,
        info: dict | None = None,
    ) -> None:
        """
        Base class for time-varying spherical harmonic coefficients.
        """
        coeffs_ndarray = np.array(coeffs).squeeze()
        epochs_ndarray = np.array(epochs).squeeze()
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
            self.coeffs_format = "cilm"
        elif coeffs_ndarray.ndim == 1:
            self.coeffs = coeffs_ndarray.copy()
            self.errors = errors_ndarray.copy() if errors is not None else None
            self.coeffs_format = "single"
        elif coeffs_ndarray.ndim in (2, 4):
            self.coeffs = coeffs_ndarray.copy()
            self.errors = errors_ndarray.copy() if errors is not None else None
            if coeffs_ndarray.ndim == 2:
                self.coeffs_format = "vector"
            else:
                self.coeffs_format = "cilm"
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

    def format_convert(self, to_format: str):
        if to_format == self.coeffs_format:
            return self.copy()
        if all(fm in ("vector", "cilm") for fm in (self.coeffs_format, to_format)):
            if to_format == "vector":
                coeffs = np.array([cilm2vector(c) for c in self.coeffs])
                errors = np.array([cilm2vector(e) for e in self.errors]) if self.errors is not None else None
            else:
                coeffs = np.array([vector2cilm(c) for c in self.coeffs])
                errors = np.array([vector2cilm(e) for e in self.errors]) if self.errors is not None else None
            return self.copy(coeffs=coeffs, errors=errors)
        else:
            msg = f"Invalid format conversion from '{self.coeffs_format}' to '{to_format}'"
            raise ValueError(msg)
        
    def resample(self, resample_epochs):
        current_epochs = self.epochs.copy()
        boolean = np.zeros_like(current_epochs, dtype=bool)
        for t in resample_epochs:
            residual = np.abs(current_epochs - t)
            if residual.min() > 0.059:
                e = current_epochs[residual.argmin()]
                msg = f"Cannot resample to '{t}' in current_epochs, the cloest is '{e}'"
                raise ValueError(msg)
            else:
                argmin = residual.argmin()
                boolean[argmin] = True
                current_epochs[argmin] = np.inf

        resample_coeffs = self.coeffs[boolean]
        resample_errors = self.errors[boolean] if self.errors is not None else None
        return self.copy(coeffs=resample_coeffs, epochs=resample_epochs, errors=resample_errors)

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
