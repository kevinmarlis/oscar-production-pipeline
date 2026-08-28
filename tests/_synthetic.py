"""Deterministic synthetic input builders shared by fixtures and generate_reference.py.

Two flavours:

* ``build_global_inputs`` — full-latitude (len == constants.y == 1440) x small-longitude
  datasets. This is the only shape ``compute_surface_currents`` accepts, because the grid
  (``y``/``um``/``vm``/``eta``) is baked into ``oscar.computation.constants`` at import time
  and ``evaluate_coriolis`` tiles ``f(y)`` (length 1440) onto the forcing arrays. Keeping the
  longitude count tiny makes the fast regression run in seconds with no committed data files.

* ``build_small_grid`` — a genuinely small (lat=N, lon=M) dataset for the grid-agnostic
  component tests (wind stress, dimensionless fields, forcing branch selection).

All fields are smooth analytic functions of latitude/longitude — no RNG — so the outputs are
byte-for-byte reproducible and the committed fingerprint is stable and diff-friendly.
"""

import numpy as np
import xarray as xr

from oscar.computation import constants as C
from oscar.computation.interp_and_grad import calculate_gradient

# A fixed reference day; value is irrelevant to the physics but keeps coords deterministic.
REF_TIME = np.array(["1993-01-01"], dtype="datetime64[ns]")


def _analytic_ssh(lat2d, lon2d):
    """Sea surface height (m), O(0.1 m), smooth in lat/lon."""
    return (
        0.15 * np.sin(np.deg2rad(2.0 * lat2d)) * np.cos(np.deg2rad(3.0 * lon2d))
        + 0.05 * np.cos(np.deg2rad(lat2d))
    )


def _analytic_sst(lat2d, lon2d):
    """Sea surface temperature (degC), warm equator / cool poles."""
    return (
        20.0
        + 8.0 * np.cos(np.deg2rad(lat2d))
        + 2.0 * np.sin(np.deg2rad(2.0 * lon2d)) * np.cos(np.deg2rad(lat2d))
    )


def _analytic_wind(lat2d, lon2d):
    """10 m wind components (m/s)."""
    u10 = 5.0 * np.cos(np.deg2rad(lat2d)) + 2.0 * np.sin(np.deg2rad(lon2d))
    v10 = 3.0 * np.sin(np.deg2rad(2.0 * lat2d)) + 1.0
    return u10, v10


def _dataset(name_to_values, lat, lon):
    data_vars = {
        name: (("time", "latitude", "longitude"), vals[np.newaxis, :, :])
        for name, vals in name_to_values.items()
    }
    return xr.Dataset(
        data_vars,
        coords={"time": REF_TIME, "latitude": lat, "longitude": lon},
    )


def build_global_inputs(nlon=8):
    """Return (ssh_grad, wind, sst_grad) on the full model latitude grid.

    * ``ssh_grad`` has ``sshx``/``sshy`` (what ``compute_dimensionless_fields`` consumes),
    * ``sst_grad`` has ``sstx``/``ssty``,
    * ``wind`` has raw ``u10``/``v10``.

    Latitude is the real ``YOSC`` grid (length 1440); longitude is the first ``nlon`` points
    of ``XOSC``. This mirrors exactly what ``main.run_compute_currents`` feeds the solver.
    """
    lat = C.YOSC
    lon = C.XOSC[:nlon]
    lon2d, lat2d = np.meshgrid(lon, lat)

    ssh = _dataset({"ssh": _analytic_ssh(lat2d, lon2d)}, lat, lon)
    sst = _dataset({"sst": _analytic_sst(lat2d, lon2d)}, lat, lon)
    u10, v10 = _analytic_wind(lat2d, lon2d)
    wind = _dataset({"u10": u10, "v10": v10}, lat, lon)

    ssh_grad = calculate_gradient(ssh, "ssh")
    sst_grad = calculate_gradient(sst, "sst")
    return ssh_grad, wind, sst_grad


def build_small_grid(nlat=7, nlon=5):
    """Small (time=1, lat=nlat, lon=nlon) inputs for grid-agnostic component tests.

    Returns a dict with ``ssh_grad``, ``sst_grad`` and ``wind`` datasets. Latitudes span the
    equator so equatorial (ff==0) behaviour is exercised.
    """
    lat = np.linspace(-30.0, 30.0, nlat)
    lon = np.linspace(0.0, 40.0, nlon)
    lon2d, lat2d = np.meshgrid(lon, lat)

    ssh = _dataset({"ssh": _analytic_ssh(lat2d, lon2d)}, lat, lon)
    sst = _dataset({"sst": _analytic_sst(lat2d, lon2d)}, lat, lon)
    u10, v10 = _analytic_wind(lat2d, lon2d)
    wind = _dataset({"u10": u10, "v10": v10}, lat, lon)

    return {
        "ssh_grad": calculate_gradient(ssh, "ssh"),
        "sst_grad": calculate_gradient(sst, "sst"),
        "wind": wind,
        "lat": lat,
        "lon": lon,
    }
