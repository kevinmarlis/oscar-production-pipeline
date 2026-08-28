import numpy as np
import pytest

from oscar.computation import constants as C
from oscar.computation.physics import get_turbulent_coefficient, get_wind_stress

# Equatorial vs global asymptotic constants baked into get_turbulent_coefficient.
A1, A2 = 8e-5, 2.85e-4
H1, H2 = 70, 81
RHOA, CDINF, WINF, K0, K1 = 1.29, 1.14e-3, 10, 0.49e-3, 0.065e-3


def test_turbulent_coefficient_shape_and_dims(wind_ds):
    w = wind_ds(3.0, 4.0)
    a, b, H = get_turbulent_coefficient(w, C.TURB_y0, C.TURB_Ly)
    for arr in (a, b, H):
        assert arr.dims == ("time", "latitude", "longitude")
        assert arr.shape == w.u10.shape


def test_turbulent_coefficient_equator_vs_high_latitude(wind_ds):
    w = wind_ds(3.0, 4.0, lat=[-60.0, -30.0, 0.0, 30.0, 60.0])
    a, b, H = get_turbulent_coefficient(w, C.TURB_y0, C.TURB_Ly)
    a_lat = a.isel(time=0, longitude=0).values
    H_lat = H.isel(time=0, longitude=0).values

    # At the equator the coefficients approach the equatorial constants...
    assert a_lat[2] == pytest.approx(A1, rel=1e-4)
    assert H_lat[2] == pytest.approx(H1, rel=1e-4)
    # ...and far from the equator they approach the global constants.
    assert a_lat[0] == pytest.approx(A2, rel=1e-6)
    assert H_lat[0] == pytest.approx(H2, rel=1e-6)


def test_turbulent_coefficient_symmetric_in_latitude(wind_ds):
    w = wind_ds(3.0, 4.0, lat=[-45.0, -20.0, 0.0, 20.0, 45.0])
    a, _, _ = get_turbulent_coefficient(w, C.TURB_y0, C.TURB_Ly)
    a_lat = a.isel(time=0, longitude=0).values
    np.testing.assert_allclose(a_lat, a_lat[::-1], rtol=1e-9)


def test_wind_stress_low_speed_uses_constant_drag(wind_ds):
    # speed = 10 is NOT > Winf(=10), so the constant Cdinf branch is used.
    w = wind_ds(8.0, 6.0)  # |wind| = 10
    tau = get_wind_stress(w)
    assert set(tau.data_vars) == {"stressu", "stressv"}
    expected_u = RHOA * CDINF * 10.0 * 8.0
    np.testing.assert_allclose(tau.stressu.values, expected_u, rtol=1e-12)


def test_wind_stress_high_speed_uses_linear_drag(wind_ds):
    w = wind_ds(9.0, 12.0)  # |wind| = 15 > 10
    tau = get_wind_stress(w)
    cd = K1 * 15.0 + K0
    expected_u = RHOA * cd * 15.0 * 9.0
    expected_v = RHOA * cd * 15.0 * 12.0
    np.testing.assert_allclose(tau.stressu.values, expected_u, rtol=1e-12)
    np.testing.assert_allclose(tau.stressv.values, expected_v, rtol=1e-12)


def test_wind_stress_preserves_dims(wind_ds):
    w = wind_ds(3.0, 4.0)
    tau = get_wind_stress(w)
    assert tau.stressu.dims == ("time", "latitude", "longitude")
