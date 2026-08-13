import numpy as np
import xarray as xr
from typing import Tuple, Dict
from .constants import *
from .physics import *


def compute_f1_forcings(
    norm: Dict[str, xr.DataArray],
    a: xr.DataArray,
    b: xr.DataArray,
    y_rad: np.ndarray,
    Av: xr.DataArray,
    H: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, np.ndarray]:
    
    grH = norm["grH"]
    grT = norm["grT"]
    tau = norm["tau"]
    wind_nd = norm["wind_nd"]

    f_poly = coriolis_polynomial(PHIm)

    ff = evaluate_coriolis(f_poly, y_rad, Av)



    P, Q = compute_complex_ekman_depth(ff, Av, H)

    F1_GrH = ssh_forcing(grH,g,Dlm,Lm,OMEGA,PHIm)

    F1_TAU0 = wind_stress_forcing(tau, P, Q, ff, wind_nd, a, b, H,TAUXm,TAUYm,rhom,um,OMEGA,PHIm,Hm,Cdm,g)

    F1_GrT = buoyancy_forcing(grT, P, Q, ff,g,Hm,DTm, Avm, Lm, um, xi)

    return F1_GrH, F1_TAU0, F1_GrT, ff


def ssh_forcing(grH: xr.DataArray, g: float, Dlm: float, Lm: float, OMEGA: float, PHIm: float) -> xr.DataArray:
    """Compute geostrophic (SSH) forcing term."""
    COEFA = g * Dlm / (2 * OMEGA * Lm * um * PHIm)

    xi = REARTH*PHIm/Lm # REARTH is implied here

    F1x = -COEFA * grH.real
    F1y = -COEFA / xi * grH.imag
    return F1x + 1j * F1y


def wind_stress_forcing(
    tau: xr.DataArray,
    P: xr.DataArray,
    Q: xr.DataArray,
    ff: np.ndarray,
    wind_nd: xr.Dataset,
    a: xr.DataArray,
    b: xr.DataArray,
    H: xr.DataArray,
    TAUXm: float,
    TAUYm: float,
    rhom: float,
    um: float,
    OMEGA: float,
    PHIm: float,
    Hm: float,
    Cdm: float,
    g: float
) -> xr.DataArray:
    """Compute wind stress forcing (Ekman effect)."""

    Av=(a/Avm)*np.fabs(np.sqrt(wind_nd.u10**2+wind_nd.v10**2)/W0)**b
    Av=Av.transpose('time','latitude','longitude')
    COEFC = TAUXm / (2 * OMEGA * rhom * Hm * PHIm * um)
    h = 30 / Hm
    mu = TAUYm / TAUXm
    BIGcrit = 6
    SMAcrit = 0.01

    TAU0x = tau.real
    TAU0y = tau.imag

    cond_IAv0=(Av==0)&(ff!=0)
    cond_If0=(ff==0)&(Av!=0)

    cond_big = ((np.abs(P) >= BIGcrit) & (np.abs(Q / 2) >= BIGcrit)) & (np.abs(P - Q / 2) >= BIGcrit)
    cond_small = ((np.abs(P) <= SMAcrit) & (np.abs(Q / 2) <= SMAcrit)) & (np.abs(P - Q / 2) <= SMAcrit)
    cond_std = (~cond_big) & (~cond_small)

    A=COEFC*2/h*np.sinh(Q/2)*np.cosh(P-Q/2)/np.sinh(P)
    F1_std=A*(TAU0x+1j*mu*TAU0y)
    F1_stdx=F1_std.real
    F1_stdy=F1_std.imag
    
    A=COEFC/h
    F1_big=A*(TAU0x+1j*mu*TAU0y)
    F1_bigx=F1_big.real
    F1_bigy=F1_big.imag
 
    A=COEFC/H
    F1_small=A*(TAU0x+1j*mu*TAU0y)
    F1_smallx=F1_small.real
    F1_smally=F1_small.imag

    F1_TAU0x =xr.where(cond_std,F1_stdx,np.nan)
    F1_TAU0y =xr.where(cond_std,F1_stdy,np.nan)

    F1_TAU0x=xr.where(cond_big,F1_bigx,F1_TAU0x)
    F1_TAU0y=xr.where(cond_big,F1_bigy,F1_TAU0y)

    F1_TAU0x=xr.where(cond_small,F1_smallx,F1_TAU0x)
    F1_TAU0y=xr.where(cond_small,F1_smally,F1_TAU0y)

    return F1_TAU0x + 1j*F1_TAU0y


def buoyancy_forcing(
    grT: xr.DataArray,
    P: xr.DataArray,
    Q: xr.DataArray,
    ff: np.ndarray,
    g: float,
    Hm: float,
    DTm: float,
    Avm: float,
    Lm: float,
    um: float,
    xi: float
) -> xr.DataArray:
    """Compute SST gradient (thermal buoyancy) forcing."""
    CHIm = thermal_expansion_coefficient(Tm, Sm)
    COEFB = g * Hm * CHIm * DTm / (2 * OMEGA * PHIm * um * Lm)  # OMEGA hardcoded for now
    h = 30 / Hm
    BIGcrit = 6
    SMAcrit = 0.01

    GrTx = grT.real
    GrTy = grT.imag

    cond_big = (np.abs(P / 2) >= BIGcrit) & (np.abs(Q / 2) >= BIGcrit) & (np.abs(P - Q / 2) >= BIGcrit)
    cond_small = (np.abs(P / 2) <= SMAcrit) & (np.abs(Q / 2) <= SMAcrit) & (np.abs(P - Q / 2) <= SMAcrit)
    cond_std = (~cond_big) & (~cond_small)

    A_std = COEFB / 2 * h / Q**2 * (4 * np.sinh((P - Q) / 2) * np.sinh(Q / 2) + Q**2 * np.cosh(P / 2)) / np.cosh(P / 2)
    F_std = A_std * (GrTx + 1j * GrTy / xi)

    A_big = COEFB * h * (1 / Q**2 + 0.5)
    F_big = A_big * (GrTx + 1j * GrTy / xi)

    A_small = COEFB * Hm / 2
    F_small = A_small * (GrTx + 1j * GrTy / xi)

    F1 = xr.where(cond_std, F_std, np.nan)
    F1 = xr.where(cond_big, F_big, F1)
    F1 = xr.where(cond_small, F_small, F1)

    return F1



