"""Shared fixtures for the OSCAR test suite.

Note on the config-at-import coupling: ``oscar.computation.constants`` reads
``io_config.yaml`` at import time and bakes the global grid (``y``, ``um``, ``vm``,
``eta``, ``spacing``) into module constants keyed off ``SSH_MODE``. Tests therefore run
against the committed default config (cmems, 0.125 deg) and monkeypatch only the specific
value a test needs.
"""

import numpy as np
import pytest
import xarray as xr

from tests import _synthetic

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
FIXTURES_DIR = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def small_grid():
    """Small (lat=7, lon=5) synthetic inputs for grid-agnostic component tests."""
    return _synthetic.build_small_grid()


@pytest.fixture
def wind_ds():
    """Factory building a (time=1, lat, lon) wind dataset with given u10/v10 fields.

    Pass scalars (broadcast over the grid) or full 2-D arrays.
    """

    def _make(u10, v10, lat=None, lon=None):
        lat = np.array([-20.0, -5.0, 0.0, 5.0, 20.0]) if lat is None else np.asarray(lat, float)
        lon = np.array([0.0, 10.0, 20.0]) if lon is None else np.asarray(lon, float)
        shape = (1, len(lat), len(lon))
        u = np.broadcast_to(np.asarray(u10, float), shape).copy()
        v = np.broadcast_to(np.asarray(v10, float), shape).copy()
        return xr.Dataset(
            {
                "u10": (("time", "latitude", "longitude"), u),
                "v10": (("time", "latitude", "longitude"), v),
            },
            coords={
                "time": _synthetic.REF_TIME,
                "latitude": lat,
                "longitude": lon,
            },
        )

    return _make


@pytest.fixture(scope="session")
def global_inputs():
    """Full-latitude x small-longitude inputs for the fast regression test (do_eq=False)."""
    return _synthetic.build_global_inputs(nlon=8)
