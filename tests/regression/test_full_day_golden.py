"""Slow golden anchor: full do_eq=True day vs committed golden fingerprint.

This exercises the expensive, most numerically sensitive path (the ``EQ`` equatorial
integral) on the real day-1 CMEMS/SST/WIND inputs. It is marked ``slow`` and skipped unless
both the golden fingerprint and the on-disk input data are present, so it runs
locally/nightly rather than on every push.

Regenerate the baseline with ``python tests/generate_reference.py`` on a trusted checkout.
"""

import warnings

import pytest

from tests import _fingerprint
from tests.generate_reference import load_real_day_inputs

pytestmark = [pytest.mark.slow, pytest.mark.regression]

RTOL = 1e-6
ATOL = 1e-9


@pytest.fixture(scope="module")
def golden(fixtures_dir):
    path = fixtures_dir / "golden_fingerprint.json"
    if not path.exists():
        pytest.skip(f"missing {path}; run `python tests/generate_reference.py` where day-1 data exists")
    return _fingerprint.load(path)


def test_full_day_matches_golden(golden):
    from oscar.computation.compute_currents import compute_surface_currents

    inputs = load_real_day_inputs()
    if inputs is None:
        pytest.skip("real day-1 input files not on disk")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssh, wind, sst = inputs
        Ug, Uw, Ub = compute_surface_currents(ssh, wind, sst, do_eq=True)

    candidate = _fingerprint.fingerprint({"Ug": Ug, "Uw": Uw, "Ub": Ub})
    problems = _fingerprint.compare(golden, candidate, rtol=RTOL, atol=ATOL)
    assert not problems, _fingerprint.format_mismatch(
        problems,
        regenerate_hint="python tests/generate_reference.py  (needs the day-1 input files on disk)",
    )
