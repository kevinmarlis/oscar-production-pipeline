import numpy as np 
from .constants import TURB_Ly, TURB_y0, y, W0, um, vm, eta, Avm
from .physics import get_turbulent_coefficient, compute_dimensionless_fields, compute_u1
from .forcing import compute_f1_forcings
from .equatorial_treatment import do_equatorial_treatment

def compute_surface_currents(ssh, wind, sst, do_eq):
    
    a, b, H = get_turbulent_coefficient(wind, TURB_y0, TURB_Ly)
    COEFG = a/Avm
    Av=COEFG*np.fabs(np.sqrt(wind.u10**2+wind.v10**2)/W0)**b
    norm = compute_dimensionless_fields(ssh, wind, sst)
    F1_GrH, F1_TAU0, F1_GrT, ff = compute_f1_forcings(norm, a, b, y, Av, H)
    U1g, U1Wh, U1Bh, U1_GrH, U1_TAU0, U1_GrT = compute_u1(y,ff, F1_GrH, F1_TAU0, F1_GrT, eta)

    print('Equatorial treatment...')
    Ug, Uw, Ub = do_equatorial_treatment(U1g, U1Wh, U1Bh, um,vm, do_eq, F1_GrH, F1_TAU0, F1_GrT, U1_GrH, U1_TAU0, U1_GrT)

    return Ug, Uw, Ub
