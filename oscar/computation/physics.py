import xarray as xr
import math
import numpy as np
from .constants import *
from typing import Dict, Tuple
from functools import lru_cache

#recompute??
def get_turbulent_coefficient(WIND: xr.Dataset, y0: float, Ly: float):

    # Constants (equatorial and global)
    a1, b1, H1 = 8e-5, 2.2, 70      # Equatorial
    a2, b2, H2 = 2.85e-4, 2.0, 81   # Global

    yy = WIND.latitude
    ones = xr.ones_like(WIND.u10)

    # Smooth transition profile based on latitude
    tanh_arg_pos = (yy - y0) * math.pi / Ly
    tanh_arg_neg = (-yy - y0) * math.pi / Ly

    def smooth_transition(c1, c2):
        A = xr.where(yy >= 0,
                     c1/2 * np.tanh(tanh_arg_pos) + c2,
                     c1/2 * np.tanh(tanh_arg_neg) + c2)
        return (A * ones).transpose("time", "latitude", "longitude")
    
    #Coefficients
    a = smooth_transition(a2 - a1, (a1 + a2) / 2)
    b = smooth_transition(b2 - b1, (b1 + b2) / 2)
    H = smooth_transition(H2 - H1, (H1 + H2) / 2)

    return a, b, H


def get_wind_stress(WIND, rhoa=1.29, Winf=10, Cdinf=1.14e-3, K0=0.49e-3, K1=0.065e-3):
   
    # Wind speed magnitude
    speed = np.sqrt(WIND.u10**2 + WIND.v10**2)

    # Drag coefficient (conditional)
    Cd = xr.where(speed > Winf, K1 * speed + K0, Cdinf)

    # # Stress components
    # stressu = rhoa * Cd * speed * WIND.u10
    # stressv = rhoa * Cd * speed * WIND.v10

    TAU=rhoa*Cd*speed*WIND
    TAU=TAU.rename(name_dict={'u10':'stressu','v10':'stressv'})

    return TAU


def compute_dimensionless_fields(
    ssh: xr.Dataset, wind: xr.Dataset, sst: xr.Dataset,
) -> Dict[str, xr.DataArray]:
    """Scales SSH, wind, and SST gradients to dimensionless form based on reference values."""

    grH = ssh.sshx/(Dlm/Lm) + 1j * ssh.sshy/(Dlm/REARTH/PHIm)

    # Wind stress (TAU0)
    TAU0In = get_wind_stress(wind)
    TAU0 = TAU0In.stressu/TAUXm + 1j * TAU0In.stressv / TAUYm

    grT = sst.sstx/(DTm/Lm) + 1j * sst.ssty/(DTm/REARTH/PHIm)

    return {
        "grH": grH,
        "grT": grT,
        "tau": TAU0,
        "wind_nd": wind / Wm
    }


def evaluate_coriolis(f_poly: np.ndarray, y: np.ndarray, Av: xr.DataArray) -> np.ndarray:
    """Evaluate polynomial approximation of Coriolis parameter at each latitude point."""
    ff = np.polyval(f_poly, y)
    nt, ny, nx = Av.shape
    ff_reshaped = np.tile(ff, (nt, nx, 1)).transpose(0, 2, 1)
    return ff_reshaped

# def coriolis_polynomial(PHIm: float) -> np.ndarray:
#     """Returns Taylor expansion polynomial for f = 2Ωsin(φ) centered at the equator."""
#     OMEGA = 7.292e-5
#     coeffs = 2 * OMEGA * np.array([1/362880, 0, -1/5040, 0, 1/120, 0, -1/6, 0, 1, 0])
#     powers = np.arange(len(coeffs), 0, -1)
#     factor = PHIm ** (powers - 2)
#     return coeffs * factor / (2 * OMEGA)


@lru_cache(maxsize=None)


def coriolis_polynomial(PHIm: float) -> np.ndarray:
    OMEGA = 7.292e-5
    coeffs = 2 * OMEGA * np.array([1/362880, 0, -1/5040, 0, 1/120, 0, -1/6, 0, 1, 0])
    powers = np.arange(len(coeffs), 0, -1)
    factor = PHIm ** (powers - 2)
    return coeffs * factor / (2 * OMEGA)


def compute_complex_ekman_depth(
    ff: np.ndarray,
    Av: xr.DataArray,
    H: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute complex Ekman layer scaling terms P and Q."""

    ff = xr.DataArray(ff, coords=Av.coords, dims=Av.dims)
    COEFD = np.sqrt(2 * OMEGA * PHIm * Hm**2 / Avm)
    h = 30 / Hm
    H = H/Hm # Assume H is already scaled


    he=np.sqrt(Av/ff/1j)
    he=xr.where((Av!=0) & (ff!=0), he, np.nan)
    P=COEFD*H/he
    Q=COEFD*h/he
    return P, Q


def compute_u1(
    y: np.ndarray,
    ff: np.ndarray,
    F1_GrH: np.ndarray,
    F1_TAU0: np.ndarray,
    F1_GrT: np.ndarray,
    eta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the outer solution (U1) from three forcing terms."""

    F1_GrH = F1_GrH.transpose('time', 'latitude', 'longitude') if F1_GrH.dims[0] != 'time' else F1_GrH


    U1_GrH = raw_velocity(y, F1_GrH, ff)
    U1g = U1_GrH.real + 1j * U1_GrH.imag / eta

    U1_TAU0 = raw_velocity(y, F1_TAU0, ff)
    U1w = U1_TAU0.real + 1j * U1_TAU0.imag / eta

    U1_GrT = raw_velocity(y, F1_GrT, ff)
    U1b = U1_GrT.real + 1j * U1_GrT.imag / eta

    return U1g, U1w, U1b, U1_GrH, U1_TAU0, U1_GrT


def raw_velocity(y: np.ndarray, F: np.ndarray, ff: np.ndarray) -> np.ndarray:
    """Compute raw solution U = F / (i f(y))."""
    with np.errstate(divide="ignore", invalid="ignore"):

        URAWx = np.where(ff != 0, F.imag / ff, 0)
        URAWy = np.where(ff != 0, -F.real / ff, 0)
    return URAWx + 1j * URAWy


def rescale_velocities(
    Ug: np.ndarray,
    Uw: np.ndarray,
    Ub: np.ndarray,
    um: float,
    vm: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rescale dimensionless currents back to dimensional values."""
    Ug_scaled = Ug.real * um + 1j * Ug.imag * vm
    Uw_scaled = Uw.real * um + 1j * Uw.imag * vm
    Ub_scaled = Ub.real * um + 1j * Ub.imag * vm
    return Ug_scaled, Uw_scaled, Ub_scaled


def combu1u2(y,ysl,ynl,U1g,U1Wh,U1Bh,U2g,U2Wh,U2Bh):
    #U1 = outer solution
    yind1=np.where(np.any([y<=ysl, y>=ynl], axis = 0))[0]
    #U2=inner solution
    yind2=np.where(np.all([y>ysl, y<ynl], axis = 0))[0]

    NaN = float("NaN")
    Ug = np.zeros(U1g.shape) + NaN*(1 + 1j)
    Uw = np.zeros(U1g.shape) + NaN*(1 + 1j)
    Ub = np.zeros(U1g.shape) + NaN*(1 + 1j)

    Ug[:,yind1,:] = U1g.real[:,yind1,:] + 1j*U1g.imag[:,yind1,:]
    Uw[:,yind1,:] = U1Wh.real[:,yind1,:] + 1j*U1Wh.imag[:,yind1,:]
    Ub[:,yind1,:] = U1Bh.real[:,yind1,:] + 1j*U1Bh.imag[:,yind1,:]

    Ug[:,yind2,:] =  U2g[:,yind2,:]
    Uw[:,yind2,:] = U2Wh[:,yind2,:]
    Ub[:,yind2,:] = U2Bh[:,yind2,:]

    return Ug, Uw, Ub