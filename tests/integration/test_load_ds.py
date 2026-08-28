"""load_ds input normalization: variable renames, longitude wrapping, unit conversion,
daily averaging. These transforms are independently verifiable, so the assertions check
correctness (e.g. 300 K with mask==1 -> 26.85 degC), not just stability.
"""

import os

import numpy as np
import pytest
import xarray as xr

from oscar.utils import data_utils as du

REF_TIME = np.array(["1993-01-01"], dtype="datetime64[ns]")


@pytest.fixture
def src_dirs(tmp_path, monkeypatch):
    ssh_final = tmp_path / "ssh_final"
    sst = tmp_path / "sst"
    wind = tmp_path / "wind"
    (ssh_final / "1993" / "01").mkdir(parents=True)
    (sst / "1993" / "01").mkdir(parents=True)
    (wind / "1993" / "01").mkdir(parents=True)
    monkeypatch.setattr(du, "SSH_SRC_FINAL_DIR", str(ssh_final))
    monkeypatch.setattr(du, "SST_SRC_DIR", str(sst))
    monkeypatch.setattr(du, "WIND_SRC_DIR", str(wind))
    monkeypatch.setattr(du, "SSH_PATTERN", "ssh_*.nc")
    monkeypatch.setattr(du, "WIND_PATTERN", "era5_*.nc")
    return ssh_final, sst, wind


def test_load_ssh_renames_adt_and_wraps_longitude(src_dirs):
    ssh_final, _, _ = src_dirs
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([-179.0, 0.0, 179.0])  # includes a negative longitude
    xr.Dataset(
        {"adt": (("time", "latitude", "longitude"), np.arange(9.0).reshape(1, 3, 3))},
        coords={"time": REF_TIME, "latitude": lat, "longitude": lon},
    ).to_netcdf(str(ssh_final / "1993" / "01" / "ssh_19930101.nc"))

    ds = du.load_ds("1993-01-01", "final", "ssh")

    assert list(ds.data_vars) == ["ssh"]  # adt -> ssh
    # longitudes wrapped to [0, 360) and sorted ascending: -179 -> 181.
    np.testing.assert_allclose(ds.longitude.values, [0.0, 179.0, 181.0])
    assert str(ds.time.values[0])[:10] == "1993-01-01"


def test_load_sst_converts_kelvin_and_applies_mask(src_dirs):
    _, sst, _ = src_dirs
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 10.0])
    xr.Dataset(
        {
            "analysed_sst": (("time", "lat", "lon"), np.array([[[300.0, 301.0], [302.0, 303.0]]])),
            "mask": (("time", "lat", "lon"), np.array([[[1, 0], [1, 1]]])),
        },
        coords={"time": REF_TIME, "lat": lat, "lon": lon},
    ).to_netcdf(str(sst / "1993" / "01" / "19930101120000-CMC-L4_GHRSST-SSTfnd-x.nc"))

    ds = du.load_ds("1993-01-01", "final", "sst")

    assert list(ds.data_vars) == ["sst"]
    assert "latitude" in ds.coords and "longitude" in ds.coords  # lat/lon renamed
    # 300 K -> 26.85 degC where mask==1; masked (mask==0) cell -> NaN.
    vals = ds.sst.values.ravel()
    np.testing.assert_allclose(vals[0], 26.85)
    assert np.isnan(vals[1])
    np.testing.assert_allclose(vals[2], 28.85)
    np.testing.assert_allclose(vals[3], 29.85)


def test_load_wind_renames_time_and_daily_averages(src_dirs):
    _, _, wind = src_dirs
    valid_time = np.array(["1993-01-01T00", "1993-01-01T12"], dtype="datetime64[ns]")
    xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), np.array([[[2.0]], [[4.0]]])),
            "v10": (("valid_time", "latitude", "longitude"), np.array([[[1.0]], [[3.0]]])),
        },
        coords={"valid_time": valid_time, "latitude": [0.0], "longitude": [0.0]},
    ).to_netcdf(str(wind / "1993" / "01" / "era5_19930101.nc"))

    ds = du.load_ds("1993-01-01", "final", "wind")

    assert set(ds.data_vars) == {"u10", "v10"}
    # two sub-daily steps averaged to one daily value.
    np.testing.assert_allclose(ds.u10.values.ravel()[0], 3.0)  # mean(2, 4)
    np.testing.assert_allclose(ds.v10.values.ravel()[0], 2.0)  # mean(1, 3)


def test_load_ds_missing_file_raises(src_dirs):
    with pytest.raises(FileNotFoundError):
        du.load_ds("1993-01-02", "final", "ssh")


def test_load_ds_unsupported_var_raises(src_dirs):
    with pytest.raises(ValueError, match="Unsupported variable"):
        du.load_ds("1993-01-01", "final", "salinity")
