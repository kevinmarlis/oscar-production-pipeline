import numpy as np
import xarray as xr

from oscar.computation import constants as C
from oscar.computation.physics import (
    compute_complex_ekman_depth,
    compute_dimensionless_fields,
)
from tests._synthetic import REF_TIME


def test_dimensionless_fields_keys_and_complex_dtype(small_grid):
    norm = compute_dimensionless_fields(
        small_grid["ssh_grad"], small_grid["wind"], small_grid["sst_grad"]
    )
    assert set(norm) == {"grH", "grT", "tau", "wind_nd"}
    for key in ("grH", "grT", "tau"):
        assert np.iscomplexobj(norm[key].values), key
        assert norm[key].dims == ("time", "latitude", "longitude")
    # wind_nd stays a Dataset of the two real components.
    assert set(norm["wind_nd"].data_vars) == {"u10", "v10"}


def test_dimensionless_grH_matches_scaling_formula(small_grid):
    ssh = small_grid["ssh_grad"]
    norm = compute_dimensionless_fields(ssh, small_grid["wind"], small_grid["sst_grad"])
    expected = ssh.sshx / (C.Dlm / C.Lm) + 1j * ssh.sshy / (C.Dlm / C.REARTH / C.PHIm)
    np.testing.assert_allclose(norm["grH"].values, expected.values, rtol=1e-12)


def test_dimensionless_propagates_nan(small_grid):
    ssh = small_grid["ssh_grad"].copy(deep=True)
    ssh["sshx"][0, 0, 0] = np.nan
    norm = compute_dimensionless_fields(ssh, small_grid["wind"], small_grid["sst_grad"])
    assert np.isnan(norm["grH"].values[0, 0, 0])


def _cell_da(values, name="x"):
    lat = np.arange(len(values), dtype=float)
    lon = np.array([0.0])
    arr = np.asarray(values, float).reshape(1, len(values), 1)
    return xr.DataArray(
        arr, dims=("time", "latitude", "longitude"),
        coords={"time": REF_TIME, "latitude": lat, "longitude": lon}, name=name,
    )


def test_ekman_depth_nan_where_av_or_ff_zero():
    # cell 0: valid; cell 1: Av==0; cell 2: ff==0 -> P,Q must be NaN in cells 1,2.
    Av = _cell_da([1.0, 0.0, 2.0])
    H = _cell_da([80.0, 80.0, 80.0])
    ff = np.array([2.0, 2.0, 0.0]).reshape(1, 3, 1)

    P, Q = compute_complex_ekman_depth(ff, Av, H)

    assert np.isfinite(P.values[0, 0, 0])
    assert np.isnan(P.values[0, 1, 0]) and np.isnan(Q.values[0, 1, 0])
    assert np.isnan(P.values[0, 2, 0]) and np.isnan(Q.values[0, 2, 0])
