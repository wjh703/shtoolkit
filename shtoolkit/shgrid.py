import numpy as np

from .shtrans import grid2cilm
from .shunit import SpharmUnit

__all__ = ['SphereGrid']

class SphereGrid():
    def __init__(
            self, 
            data: np.ndarray,
            epochs: np.ndarray, 
            unit: SpharmUnit,
            errors: np.ndarray | None = None, 
            error_kind: str | None = None,
            name: str | None = None, 
        ) -> None:
        nlat= data.shape[-2]
        max_resol = nlat//2-1

        self.data = data
        self.epochs = epochs
        self.unit: SpharmUnit = unit
        self.max_resol = max_resol
        self.errors = errors
        self.error_kind = error_kind
        self.name = name
    
    def expand(self, lmax_calc: int | None = None):
        from .shcoeff import SpharmCoeff
        if lmax_calc is not None:
            coeffs = np.array([grid2cilm(grid, lmax_calc) for grid in self.data])
        else:
            coeffs = np.array([grid2cilm(grid, self.max_resol) for grid in self.data])
        return SpharmCoeff(coeffs, self.epochs.copy(), self.unit)