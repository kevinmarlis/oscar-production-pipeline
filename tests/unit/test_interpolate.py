import numpy as np
import pytest
import xarray as xr

from oscar.computation import interp_and_grad
from oscar.computation.interp_and_grad import interpolate_dataset
from tests._synthetic import REF_TIME


def _coarse_dataset():
    lat = np.arange(-80.0, 81.0, 20.0)
    lon = np.arange(0.0, 360.0, 20.0)
    data = np.random.default_rng(0).standard_normal((1, len(lat), len(lon)))
    return xr.Dataset(
        {"ssh": (("time", "latitude", "longitude"), data)},
        coords={"time": REF_TIME, "latitude": lat, "longitude": lon},
    )


def test_interpolate_to_cmems_grid_shape():
    ds = _coarse_dataset()
    out = interpolate_dataset(ds)
    # cmems target grid: 0.125 deg spacing
    expected_lon = np.arange(0, 360, 0.125)
    expected_lat = np.arange(-89.9375, 90.0625, 0.125)
    np.testing.assert_allclose(out.longitude.values, expected_lon)
    np.testing.assert_allclose(out.latitude.values, expected_lat)
    assert out.ssh.shape == (1, len(expected_lat), len(expected_lon))


def test_interpolate_unknown_mode_raises(monkeypatch):
    monkeypatch.setattr(interp_and_grad, "SSH_MODE", "bogus")
    with pytest.raises(ValueError, match="Unknown SSH source"):
        interpolate_dataset(_coarse_dataset())
