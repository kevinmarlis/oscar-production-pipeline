"""write_oscar output round-trip: path construction and field mapping.

Verifies the file lands at <OUTPUTDIR>/<year>/<month>/<prefix><YYYYMMDD>.nc, that total
currents are the sum of the three components, and that the geostrophic/Ekman components are
written from the real/imag parts. Values are compared with a float32-encoding tolerance.
"""

import os

import numpy as np
import xarray as xr

from oscar.utils import data_utils as du

# to_netcdf encodes data vars as float32, so round-trip comparisons use this tolerance.
F32_ATOL = 1e-5


def _ref_dataset(lat, lon):
    return xr.Dataset(
        {"ssh": (("time", "latitude", "longitude"), np.zeros((1, len(lat), len(lon))))},
        coords={
            "time": np.array(["1993-01-01"], dtype="datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_write_oscar_round_trip(tmp_path):
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0])
    ref = _ref_dataset(lat, lon)

    rng = np.random.default_rng(0)

    def cx():
        return 0.1 * (rng.standard_normal((1, 3, 4)) + 1j * rng.standard_normal((1, 3, 4)))

    Ug, Uw, Ub = cx(), cx(), cx()

    du.write_oscar(
        ref, Ug, Uw, Ub,
        "oscar_currents_final_", str(tmp_path),
        "SSH", "WIND", "SST", "OSCAR", "SUMMARY", "OID", "DOI", "cmems",
    )

    out_path = tmp_path / "1993" / "01" / "oscar_currents_final_19930101.nc"
    assert out_path.exists()

    out = xr.open_dataset(out_path)
    try:
        # ub/vb are commented out in write_metadata; document that the buoyancy component
        # is intentionally not written to the product.
        assert set(out.data_vars) == {"u", "v", "ug", "vg", "uw", "vw"}
        assert out.sizes["time"] == 1

        total = Ug + Uw + Ub
        u_file = out.u.transpose("time", "latitude", "longitude").values
        v_file = out.v.transpose("time", "latitude", "longitude").values
        ug_file = out.ug.transpose("time", "latitude", "longitude").values
        uw_file = out.uw.transpose("time", "latitude", "longitude").values

        np.testing.assert_allclose(u_file, total.real, atol=F32_ATOL)
        np.testing.assert_allclose(v_file, total.imag, atol=F32_ATOL)
        np.testing.assert_allclose(ug_file, Ug.real, atol=F32_ATOL)
        np.testing.assert_allclose(uw_file, Uw.real, atol=F32_ATOL)
    finally:
        out.close()


def test_write_oscar_overwrites_existing(tmp_path):
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    ref = _ref_dataset(lat, lon)
    zeros = np.zeros((1, 2, 2), dtype=complex)

    out_path = tmp_path / "1993" / "01" / "oscar_currents_final_19930101.nc"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"stale")  # pre-existing file must be replaced, not appended.

    du.write_oscar(
        ref, zeros, zeros, zeros,
        "oscar_currents_final_", str(tmp_path),
        "SSH", "WIND", "SST", "OSCAR", "SUMMARY", "OID", "DOI", "cmems",
    )

    out = xr.open_dataset(out_path)
    try:
        assert "u" in out.data_vars
    finally:
        out.close()
