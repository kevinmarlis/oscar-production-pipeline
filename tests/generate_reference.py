#!/usr/bin/env python
"""Regenerate the committed regression fingerprints from the current (trusted) code.

Run this once from a trusted checkout to establish the baseline, and re-run it (with
review of the JSON diff) whenever a change is a *deliberate* numerical improvement.

Two fingerprints are produced:

* ``fixtures/reference_fingerprint.json`` — FAST. Built from the deterministic synthetic
  global-latitude / small-longitude inputs with ``do_eq=False``. Reproducible anywhere,
  no external data, runs in CI.

* ``fixtures/golden_fingerprint.json`` — SLOW. Built from the real day-1 CMEMS/SST/WIND
  input files on disk with ``do_eq=True`` (the expensive equatorial integral). Only
  regenerated when that data is present; skipped otherwise.

Usage:
    python tests/generate_reference.py            # both if data present
    python tests/generate_reference.py --fast     # fast fingerprint only
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import _fingerprint, _synthetic  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_DATE = "1993-01-01"


def build_fast_fingerprint():
    from oscar.computation.compute_currents import compute_surface_currents

    ssh, wind, sst = _synthetic.build_global_inputs(nlon=8)
    Ug, Uw, Ub = compute_surface_currents(ssh, wind, sst, do_eq=False)
    return _fingerprint.fingerprint({"Ug": Ug, "Uw": Uw, "Ub": Ub})


def load_real_day_inputs(date=GOLDEN_DATE, oscar_mode="final"):
    """Reproduce main.run_compute_currents' input preparation for one day.

    Returns (ssh_grad, wind, sst_grad) ready for compute_surface_currents, or None if the
    required input files are not on disk.
    """
    from oscar.computation.interp_and_grad import calculate_gradient, interpolate_dataset
    from oscar.utils.data_utils import load_ds

    try:
        ssh = interpolate_dataset(load_ds(date, oscar_mode, var="ssh"))
        ssh = calculate_gradient(ssh, "ssh")
        sst = interpolate_dataset(load_ds(date, oscar_mode, var="sst"))
        sst = calculate_gradient(sst, "sst")
        wind = interpolate_dataset(load_ds(date, oscar_mode, var="wind"))
    except FileNotFoundError as exc:
        print(f"  golden inputs not available: {exc}")
        return None
    return ssh, wind, sst


def build_golden_fingerprint():
    from oscar.computation.compute_currents import compute_surface_currents

    inputs = load_real_day_inputs()
    if inputs is None:
        return None
    ssh, wind, sst = inputs
    print("  running full do_eq=True day (this is slow)...")
    Ug, Uw, Ub = compute_surface_currents(ssh, wind, sst, do_eq=True)
    return _fingerprint.fingerprint({"Ug": Ug, "Uw": Uw, "Ub": Ub})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="only regenerate the fast fingerprint")
    args = parser.parse_args()

    FIXTURES.mkdir(exist_ok=True)

    print("Building FAST fingerprint (synthetic inputs, do_eq=False)...")
    fast = build_fast_fingerprint()
    fast_path = FIXTURES / "reference_fingerprint.json"
    _fingerprint.save(fast, fast_path)
    print(f"  wrote {fast_path.relative_to(REPO_ROOT)}")

    if not args.fast:
        print("Building GOLDEN fingerprint (real day-1 inputs, do_eq=True)...")
        golden = build_golden_fingerprint()
        if golden is not None:
            golden_path = FIXTURES / "golden_fingerprint.json"
            _fingerprint.save(golden, golden_path)
            print(f"  wrote {golden_path.relative_to(REPO_ROOT)}")
        else:
            print("  skipped golden fingerprint (no input data on disk)")


if __name__ == "__main__":
    main()
