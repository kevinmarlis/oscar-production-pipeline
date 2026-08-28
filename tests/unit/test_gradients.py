"""Analytic checks on calculate_gradient.

The gradient converts degrees to metres with ``ddegdm = 180 / (pi * REARTH)`` using the
canonical ``REARTH`` from ``oscar.computation.constants`` (single source of truth). The
x-gradient is additionally divided by ``cos(lat)``. Building fields that are exact linear
ramps makes the expected gradient a known constant, so any sign/scaling regression is caught
exactly.
"""

import math

import numpy as np
import xarray as xr

from oscar.computation.constants import REARTH
from oscar.computation.interp_and_grad import calculate_gradient

DDEGDM = 180.0 / (math.pi * REARTH)


def _field(values_2d, lat, lon):
    return xr.Dataset(
        {"ssh": (("time", "latitude", "longitude"), values_2d[np.newaxis, :, :])},
        coords={
            "time": np.array(["1993-01-01"], dtype="datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_longitude_ramp_gives_cos_scaled_x_gradient():
    lat = np.array([-30.0, 0.0, 30.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0])
    c = 0.5
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds = _field(c * lon2d, lat, lon)

    grad = calculate_gradient(ds, "ssh")

    # d(c*lon)/dlon = c, then scaled by ddegdm / cos(lat)
    expected_x = c * DDEGDM / np.cos(np.deg2rad(lat))
    got_x = grad.sshx.isel(time=0, longitude=1).values
    np.testing.assert_allclose(got_x, expected_x, rtol=1e-12)
    # A pure longitude ramp has no latitude gradient.
    assert np.nanmax(np.abs(grad.sshy.values)) == 0.0


def test_latitude_ramp_gives_constant_y_gradient():
    lat = np.array([-30.0, 0.0, 30.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0])
    c = 0.5
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds = _field(c * lat2d, lat, lon)

    grad = calculate_gradient(ds, "ssh")

    # d(c*lat)/dlat = c, scaled by ddegdm (no cos factor on y).
    expected_y = c * DDEGDM
    np.testing.assert_allclose(grad.sshy.values, expected_y, rtol=1e-12)
    assert np.nanmax(np.abs(grad.sshx.values)) == 0.0


def test_gradient_dims_are_time_lat_lon():
    lat = np.array([-10.0, 0.0, 10.0])
    lon = np.array([0.0, 5.0, 10.0])
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds = _field(lat2d + lon2d, lat, lon)

    grad = calculate_gradient(ds, "ssh")

    assert grad.sshx.dims == ("time", "latitude", "longitude")
    assert grad.sshy.dims == ("time", "latitude", "longitude")
    assert set(grad.data_vars) == {"sshx", "sshy"}
