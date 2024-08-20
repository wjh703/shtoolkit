import re
from typing import Literal, TypedDict
import numpy as np


SH_CONST = {
    'gave': 9.81,
    'rho_i': 934,
    'rho_w' : 1000,
    'rho_e': 5517,
    'a': 6.371e6,
    'm_e': 5.976e24,
    'k2e_t': 0.298,
    'h2e_t': 0.604,
    'k2e': -0.305,
    'klf': 0.942,
    'omega': 7.292e-5,
    'cmina': 2.634e35,
    'moi_a': 8.0077e37,
    'moi_c': 8.0345e37,  # Polar moment of inertia
    'moi_cm': 7.1236e37,
    'kan': -0.0011,
    'cw_freq': 1.649e-7 # Chandler wobble frequency (elastic earth)
}


SpharmUnit = Literal['mmewh', 'mewh', 'kmewh',
                     'mmgeo', 'mgeo', 'kmgeo', 
                     'mmupl', 'mupl', 'kmupl',
                     'kgm2mass', 'stokes']


class LoadLoveNumDict(TypedDict):
    h_el: np.ndarray 
    l_el: np.ndarray
    k_el: np.ndarray


def convert(
        coeffs: np.ndarray,
        current_unit: SpharmUnit,
        new_unit: SpharmUnit,
        lln: LoadLoveNumDict | None = None,
        errors: np.ndarray | None = None
    ):
    lmax = coeffs.shape[-2]-1
    coeffs_current = coeffs.copy()

    unit_regex = r'kgm2|mass|mm?|km|ewh|geo|upl|stokes'
    unit_current = re.findall(unit_regex, current_unit)
    unit_new = re.findall(unit_regex, new_unit)

    if len(unit_current) == 2:
        dimens_current, kind_current = unit_current
        if dimens_current == 'mm':
            coeffs_current /= 1000
        elif dimens_current == 'km':
            coeffs_current *= 1000
    else:
        kind_current = unit_current[0]

    if len(unit_new) == 2:
        dimens_new, kind_new = unit_new
    else:
        kind_new = unit_new[0]
        dimens_new = None

    func_name = kind_current + '2' + kind_new

    if func_name in func_without_lln:
        coef = ConvertCoefDict[func_name]
    elif func_name in func_with_lln:
        if lln is None:
            msg = "Lack of input value 'lln', which should define as a LoadLoveNumDict."
            raise ValueError(msg)
        coef = ConvertCoefDict[func_name](lmax, lln) # type: ignore
    elif func_name in func_without_change:
        coef = 1
    else:
        msg = "Invalid unit for match 'ConvertCoefDict', check if both 'current_unit' and 'new_unit' are 'SpharmUnit"    
        raise ValueError(msg)
    
    coeffs_new = coeffs_current*coef
    if dimens_new == 'mm':
        coeffs_new *= 1000
    elif dimens_new == 'km':
        coeffs_new /= 1000
    return coeffs_new

def ewh2geo(
        lmax: int, 
        lln: LoadLoveNumDict   
    ) -> np.ndarray:
    k_el = lln['k_el']
    rho_w = SH_CONST['rho_w']
    rho_e = SH_CONST['rho_e']
    e2g = np.zeros((lmax + 1, lmax + 1))
    for i in range(lmax + 1):
        j = np.arange(i + 1)
        e2g[i, j] = 3.0 * rho_w / rho_e * (1.0 + k_el[i]) / (2.0 * i + 1.0)
    coef = np.array([e2g, e2g])
    return coef

def ewh2stokes(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = ewh2geo(lmax, lln)
    coef[coef.nonzero()] /= SH_CONST['a']
    return coef


def ewh2upl(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    h_el = lln['h_el']
    rho_w = SH_CONST['rho_w']
    rho_e = SH_CONST['rho_e']
    e2u = np.zeros((lmax + 1, lmax + 1))
    for i in range(lmax + 1):
        j = np.arange(i + 1)
        e2u[i, j] = 3.0 * rho_w / rho_e * h_el[i] / (2.0 * i + 1.0)
    coef = np.array([e2u, e2u])
    return coef


def geo2ewh(
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = ewh2geo(lmax, lln)
    coef[coef.nonzero()] = np.reciprocal(coef[coef.nonzero()])
    return coef

def geo2mass(
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = geo2ewh(lmax, lln)
    coef[coef.nonzero()] *= SH_CONST['rho_w']
    return coef
   
def geo2upl(
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    k_el = lln['k_el']
    h_el = lln['h_el']
    g2u = np.zeros((lmax + 1, lmax + 1))
    for i in range(lmax + 1):
        j = np.arange(i + 1)
        g2u[i, j] =  h_el[i] / (1.0 + k_el[i])
    coef = np.array([g2u, g2u])
    return coef

def upl2ewh(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = ewh2upl(lmax, lln)
    coef[coef.nonzero()] = np.reciprocal(coef[coef.nonzero()])
    return coef

def upl2mass(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = upl2ewh(lmax, lln)
    coef[coef.nonzero()] *= SH_CONST['rho_w']
    return coef

def upl2geo(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = geo2upl(lmax, lln)
    coef[coef.nonzero()] = np.reciprocal(coef[coef.nonzero()])
    return coef

def upl2stokes(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = upl2geo(lmax, lln)
    coef[coef.nonzero()] /= SH_CONST['a']
    return coef


def stokes2ewh(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = ewh2stokes(lmax, lln)
    coef[coef.nonzero()] = np.reciprocal(coef[coef.nonzero()])
    return coef

def stokes2mass(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = stokes2ewh(lmax, lln)
    coef[coef.nonzero()] *= SH_CONST['rho_w']
    return coef

def stokes2upl(        
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = upl2stokes(lmax, lln)
    coef[coef.nonzero()] = np.reciprocal(coef[coef.nonzero()])
    return coef

def mass2geo(
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = ewh2geo(lmax, lln)
    coef[coef.nonzero()] /= SH_CONST['rho_w']
    return coef

def mass2upl(
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = ewh2upl(lmax, lln)
    coef[coef.nonzero()] /= SH_CONST['rho_w']
    return coef

def mass2stokes(
        lmax: int, 
        lln: LoadLoveNumDict
    ) -> np.ndarray:
    coef = stokes2mass(lmax, lln)
    coef[coef.nonzero()] = np.reciprocal(coef[coef.nonzero()])
    return coef

ConvertCoefDict = {
    'ewh2geo': ewh2geo,
    'ewh2upl': ewh2upl,
    'ewh2stokes': ewh2stokes,
    'ewh2mass': SH_CONST['rho_w'],
    'geo2ewh': geo2ewh,
    'geo2upl': geo2upl, 
    'geo2mass': geo2mass,
    'geo2stokes': 1/SH_CONST['a'],
    'upl2geo': upl2geo,
    'upl2ewh': upl2ewh,
    'upl2mass': upl2mass,
    'upl2stokes': upl2stokes,
    'mass2ewh': 1/SH_CONST['rho_w'],
    'mass2geo': mass2geo,
    'mass2upl': mass2upl,
    'mass2stokes': mass2stokes,
    'stokes2ewh': stokes2ewh,
    'stokes2geo': SH_CONST['a'],
    'stokes2upl': stokes2upl,
    'stokes2mass': stokes2mass
}

func_without_lln = set(['ewh2mass', 'mass2ewh', 'geo2stokes', 'stokes2geo'])
func_with_lln = set(ConvertCoefDict.keys())-func_without_lln
func_without_change = set(['ewh2ewh', 'geo2geo', 'upl2upl', 'mass2mass', 'stokes2stokes'])