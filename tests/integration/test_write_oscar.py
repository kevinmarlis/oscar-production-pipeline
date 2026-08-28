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


# --- metadata contract ------------------------------------------------
#
# The output product currently records its provenance in *attributes*, and the
# FINAL/INTERIM identity is passed in per-mode by main.run_compute_currents. The planned
# output-stream consolidation (CONSOLIDATE_OUTPUT_STREAMS_PLAN.md) moves provenance from the
# path/filename into metadata (single identity + ssh_source_type/dataset/doi attrs). These
# tests lock the *current* metadata contract so that refactor can diff deliberate changes
# (id/doi/source) against accidental ones (units/standard names/bounds).

def _write_with_identity(tmp_path, ssh_desc, wind_desc, sst_desc,
                         long_desc, summary, oscar_id, doi, ssh_mode="cmems"):
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0])
    ref = _ref_dataset(lat, lon)
    z = np.zeros((1, 3, 4), dtype=complex)
    du.write_oscar(
        ref, z, z, z, "oscar_currents_final_", str(tmp_path),
        ssh_desc, wind_desc, sst_desc, long_desc, summary, oscar_id, doi, ssh_mode,
    )
    return xr.open_dataset(tmp_path / "1993" / "01" / "oscar_currents_final_19930101.nc")


def test_write_oscar_identity_args_land_in_attributes(tmp_path):
    out = _write_with_identity(
        tmp_path,
        ssh_desc="SSHDESC", wind_desc="WINDDESC", sst_desc="SSTDESC",
        long_desc="OSCARLONG", summary="OSCARSUMM",
        oscar_id="OSCAR_L4_OC_FINAL_V3.0", doi="10.5067/OSCAR-25F20",
    )
    try:
        g = out.attrs
        # product identity (per-mode today; single identity after consolidation)
        assert g["id"] == "OSCAR_L4_OC_FINAL_V3.0"
        assert g["title"] == "OSCARLONG"
        assert g["summary"].endswith("OSCARSUMM")
        assert "10.5067/OSCAR-25F20" in g["references"]
        # provenance-carrying source strings (this is what moves to dedicated attrs)
        for desc in ("SSHDESC", "WINDDESC", "SSTDESC"):
            assert desc in g["source"]
        # per-variable provenance: total u/v cite all inputs; component vars cite only theirs.
        for desc in ("SSHDESC", "WINDDESC", "SSTDESC"):
            assert desc in out.u.attrs["source"]
        assert out.ug.attrs["source"] == "SSH source: SSHDESC"
        assert out.uw.attrs["source"] == "WIND source: WINDDESC"
    finally:
        out.close()


def test_write_oscar_cf_metadata_contract(tmp_path):
    out = _write_with_identity(
        tmp_path,
        ssh_desc="S", wind_desc="W", sst_desc="T",
        long_desc="L", summary="SUM",
        oscar_id="OID", doi="DOI",
    )
    try:
        g = out.attrs
        # CF / ACDD structural attributes downstream tooling relies on.
        assert g["Conventions"].startswith("CF-1.8")
        assert g["standard_name_vocabulary"] == "NetCDF Climate and Forecast (CF) Metadata Convention"
        assert g["processing_level"] == "L4"
        assert g["product_version"] == "v3.0"
        assert g["geospatial_lat_resolution"] == "0.125 degree"  # cmems
        assert g["time_coverage_start"] == "1993-01-01T00:00:00"
        assert g["time_coverage_end"] == "1993-01-01T23:59:59"

        # per-variable physical contract.
        for var in ("u", "v", "ug", "vg", "uw", "vw"):
            assert out[var].attrs["units"] == "m s-1"
            assert out[var].attrs["valid_min"] == -3.0
            assert out[var].attrs["valid_max"] == 3.0
        assert out.u.attrs["standard_name"] == "eastward_sea_water_velocity"
        assert out.v.attrs["standard_name"] == "northward_sea_water_velocity"
        assert out.ug.attrs["standard_name"] == "geostrophic_eastward_sea_water_velocity"
    finally:
        out.close()


def test_write_oscar_neurost_resolution(tmp_path):
    # ssh_mode drives the advertised grid resolution in global attrs.
    out = _write_with_identity(
        tmp_path,
        ssh_desc="S", wind_desc="W", sst_desc="T",
        long_desc="L", summary="SUM", oscar_id="OID", doi="DOI",
        ssh_mode="neurost",
    )
    try:
        assert out.attrs["geospatial_lat_resolution"] == "0.1 degree"
    finally:
        out.close()
