import math
import numpy as np
from ..config.setup import SSH_MODE

def thermal_expansion_coefficient(T: float, S: float) -> float:
    """Returns thermal expansion coefficient of seawater (1/K) using Gill’s equation."""
    R = np.array([6.536332e-9, -1.120083e-6, 1.001685e-4,
                  -9.095290e-3, 6.793952e-2, 999.842594])
    Q = np.array([5.3875e-9, -8.2467e-7, 7.6438e-5,
                  -4.0899e-3, 0.824493])
    N = np.array([-1.6546e-6, 1.0227e-4, -5.72466e-3])
    K = 4.8314e-4

    # Polynomial expansion
    p1 = np.append(0, Q)
    p1_5 = np.append(np.zeros(3), N)
    p2 = np.append(np.zeros(5), K)

    rhoc = S * p1 + S**1.5 * p1_5 + S**2 * p2
    rho_poly = R + rhoc
    rho = np.polyval(rho_poly, T)
    drho_dT = np.polyval(np.polyder(rho_poly), T)

    chi = -drho_dT / rho
    return chi

LATITUDE_COEFFICIENT_CACHE = {}
REARTH = 6371e3
OMEGA = 7.292e-5
g = 9.8
rhom = 1025
rhoam = 1.2
Cdm = 1.4e-3
um, vm = 0.4, 0.05
Um = (um**2 + vm**2)**0.5
Lm = 4e5
Cm = 3
BETA = 2 * OMEGA / REARTH
Ro = (Cm / (2 * BETA))**0.5
PHIm = Ro / REARTH
Hm = 30
Dlm = 0.08
WXm, WYm = 5, 1
Wm = math.sqrt(WXm**2 + WYm**2)
TAUXm = rhoam * Cdm * WXm * Wm
TAUYm = rhoam * Cdm * WYm * Wm
TAUm = (TAUXm**2 + TAUYm**2)**0.5
USTARm = (TAUm / rhom)**0.5
Avm = TAUm / (rhom * Um / Hm)
Tm = 26
Sm = 35
DTm = 2
W0 = 1
h = 30
thetam = 1 / (2 * OMEGA * PHIm)
NORDEREF = 5
PREC = 1e-6
BIGcrit = 6
SMAcrit = 0.01
TURB_y0 = 12
TURB_Ly = 5

# Polynomial expansion of Coriolis parameter f
f = 2 * OMEGA * np.array([1/362880, 0, -1/5040, 0, 1/120, 0, -1/6, 0, 1, 0])
powers = np.arange(len(f), 0, -1)
FACTOR = PHIm**(powers - 2)
f = f * FACTOR / (2 * OMEGA)


if SSH_MODE == "cmems":
    spacing = 0.125
elif SSH_MODE == "neurost":
    spacing = 0.1 


XOSC = np.arange(0, 360, spacing)
YOSC = np.arange(-89.9375, 90.0625, spacing)

y = YOSC * math.pi / 180

xi = REARTH * PHIm / Lm
eta = vm / um


y = y / PHIm
CHIm = thermal_expansion_coefficient(Tm, Sm)


