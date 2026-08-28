"""Branch-selection tests for the three forcing regimes.

Both wind_stress_forcing and buoyancy_forcing pick a formula per grid cell based on the
Ekman scaling terms P and Q relative to BIGcrit(=6) / SMAcrit(=0.01). We build a 3-cell
column that lands one cell in each regime and assert the output equals the independently
recomputed formula for that regime.

Care: for wind stress the "std" formula collapses onto the "big" formula whenever P == Q
(because 2*sinh(x)cosh(x) = sinh(2x)), so the std cell uses P != Q.
"""

import numpy as np
import xarray as xr

from oscar.computation import constants as C
from oscar.computation.forcing import buoyancy_forcing, wind_stress_forcing
from tests._synthetic import REF_TIME

LAT = np.array([-5.0, 0.0, 5.0])  # cells: big, small, std
LON = np.array([0.0])


def _col(values):
    arr = np.asarray(values).reshape(1, 3, 1)
    return xr.DataArray(
        arr, dims=("time", "latitude", "longitude"),
        coords={"time": REF_TIME, "latitude": LAT, "longitude": LON},
    )


def _complex_col(values):
    values = np.asarray(values)
    return _col(values.real.astype(float)) + 1j * _col(values.imag.astype(float))


def test_wind_stress_forcing_branch_selection():
    # big: P=20,Q=16 (|P|,|Q/2|,|P-Q/2| all >=6); small: 0.005; std: P=1.0,Q=0.6 (P!=Q).
    P = _complex_col([20.0 + 0j, 0.005 + 0j, 1.0 + 0j])
    Q = _complex_col([16.0 + 0j, 0.005 + 0j, 0.6 + 0j])
    tau = _complex_col([2 + 3j, 2 + 3j, 2 + 3j])
    wind_nd = xr.Dataset({"u10": _col([1.0, 1.0, 1.0]), "v10": _col([1.0, 1.0, 1.0])})
    a = _col([2e-4, 2e-4, 2e-4])
    b = _col([2.0, 2.0, 2.0])
    H = _col([80.0, 80.0, 80.0])
    ff = np.array([2.0, 2.0, 2.0]).reshape(1, 3, 1)

    F = wind_stress_forcing(
        tau, P, Q, ff, wind_nd, a, b, H,
        C.TAUXm, C.TAUYm, C.rhom, C.um, C.OMEGA, C.PHIm, C.Hm, C.Cdm, C.g,
    ).values.ravel()

    COEFC = C.TAUXm / (2 * C.OMEGA * C.rhom * C.Hm * C.PHIm * C.um)
    h = 30 / C.Hm
    mu = C.TAUYm / C.TAUXm
    tx, ty = 2.0, 3.0
    stress = tx + 1j * mu * ty

    exp_big = COEFC / h * stress
    exp_small = COEFC / 80.0 * stress
    Pv, Qv = 1.0, 0.6
    exp_std = COEFC * 2 / h * np.sinh(Qv / 2) * np.cosh(Pv - Qv / 2) / np.sinh(Pv) * stress

    np.testing.assert_allclose(F[0], exp_big, rtol=1e-12)
    np.testing.assert_allclose(F[1], exp_small, rtol=1e-12)
    np.testing.assert_allclose(F[2], exp_std, rtol=1e-12)


def test_wind_stress_forcing_nan_when_P_is_nan():
    # If P is NaN, no regime condition is met and the field falls through to NaN.
    def one(v):
        return xr.DataArray(
            np.array(v).reshape(1, 1, 1), dims=("time", "latitude", "longitude"),
            coords={"time": REF_TIME, "latitude": [0.0], "longitude": [0.0]},
        )

    P = one(np.nan + 0j)
    Q = one(1.0 + 0j)
    tau = one(2 + 3j)
    wind_nd = xr.Dataset({"u10": one(1.0), "v10": one(1.0)})
    a = one(2e-4)
    b = one(2.0)
    H = one(80.0)
    ff = np.array([2.0]).reshape(1, 1, 1)

    F = wind_stress_forcing(
        tau, P, Q, ff, wind_nd, a, b, H,
        C.TAUXm, C.TAUYm, C.rhom, C.um, C.OMEGA, C.PHIm, C.Hm, C.Cdm, C.g,
    ).values.ravel()
    assert np.isnan(F[0])


def test_buoyancy_forcing_branch_selection():
    # buoyancy conditions use |P/2|,|Q/2|,|P-Q/2|; P=Q is fine here (distinct formulas).
    P = _complex_col([20.0 + 0j, 0.005 + 0j, 1.0 + 0j])
    Q = _complex_col([20.0 + 0j, 0.005 + 0j, 1.0 + 0j])
    grT = _complex_col([1 + 1j, 1 + 1j, 1 + 1j])
    ff = np.array([2.0, 2.0, 2.0]).reshape(1, 3, 1)

    F = buoyancy_forcing(
        grT, P, Q, ff, C.g, C.Hm, C.DTm, C.Avm, C.Lm, C.um, C.xi
    ).values.ravel()

    COEFB = C.g * C.Hm * C.CHIm * C.DTm / (2 * C.OMEGA * C.PHIm * C.um * C.Lm)
    h = 30 / C.Hm
    gterm = 1.0 + 1j * 1.0 / C.xi

    exp_big = COEFB * h * (1 / 20.0**2 + 0.5) * gterm
    exp_small = COEFB * C.Hm / 2 * gterm
    Pv = Qv = 1.0
    A_std = COEFB / 2 * h / Qv**2 * (
        4 * np.sinh((Pv - Qv) / 2) * np.sinh(Qv / 2) + Qv**2 * np.cosh(Pv / 2)
    ) / np.cosh(Pv / 2)
    exp_std = A_std * gterm

    np.testing.assert_allclose(F[0], exp_big, rtol=1e-12)
    np.testing.assert_allclose(F[1], exp_small, rtol=1e-12)
    np.testing.assert_allclose(F[2], exp_std, rtol=1e-12)
