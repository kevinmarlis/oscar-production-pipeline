"""Fast regression anchor: compute_surface_currents(do_eq=False) vs committed fingerprint.

This is the primary refactor safety net that runs in CI. It rebuilds the deterministic
synthetic global-latitude inputs, runs the cheap (non-equatorial) solver path, and asserts
the per-field fingerprint matches the committed baseline within tolerance. Any silent
numerical drift in the vectorized physics upstream of the equatorial integral trips it.

Regenerate the baseline with ``python tests/generate_reference.py`` when a change is a
deliberate, reviewed numerical improvement.
"""

import warnings

import pytest

from tests import _fingerprint, _synthetic

pytestmark = pytest.mark.regression

# Fingerprint tolerances: tight enough to catch drift, loose enough to absorb
# platform/BLAS rounding differences in the vectorized ops.
RTOL = 1e-6
ATOL = 1e-9


@pytest.fixture(scope="module")
def reference(fixtures_dir):
    path = fixtures_dir / "reference_fingerprint.json"
    if not path.exists():
        pytest.skip(f"missing {path}; run `python tests/generate_reference.py --fast`")
    return _fingerprint.load(path)


def test_fast_compute_matches_fingerprint(reference):
    from oscar.computation.compute_currents import compute_surface_currents

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssh, wind, sst = _synthetic.build_global_inputs(nlon=8)
        Ug, Uw, Ub = compute_surface_currents(ssh, wind, sst, do_eq=False)

    candidate = _fingerprint.fingerprint({"Ug": Ug, "Uw": Uw, "Ub": Ub})
    problems = _fingerprint.compare(reference, candidate, rtol=RTOL, atol=ATOL)
    assert not problems, _fingerprint.format_mismatch(
        problems, regenerate_hint="python tests/generate_reference.py --fast"
    )
