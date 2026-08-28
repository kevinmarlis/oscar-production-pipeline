# OSCAR test suite

A safety net to lock the current (trusted) numerical behaviour before refactoring. The
primary refactoring risk for this scientific code is *silent numerical drift*, so the suite
combines fast correctness tests with a compact regression fingerprint.

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
pytest -m "not slow"   # unit + component + fast regression + download mocks (CI tier, seconds)
pytest -m slow         # full-day do_eq=True golden regression (needs day-1 data, ~75s)
pytest                 # everything
```

## Layout

Organized by the classic isolation axis: unit (isolated, in-memory) / integration
(filesystem + mocked network) / regression (whole-pipeline behavior anchor).

- `unit/` — isolated, in-memory tests with no I/O:
  - pure scalar/array functions (date helpers, thermal expansion, gradients on analytic
    fields, coriolis/velocity algebra, `find_pieces`);
  - vectorized field physics on small synthetic xarray inputs (wind stress, dimensionless
    fields, Ekman-depth NaN handling, forcing regime-branch selection, interpolation shape).
- `integration/` — single functions exercised against the real filesystem (and mocked
  network). This is the orchestration glue a structural refactor is most likely to churn,
  and it is *not* a black box — assertions verify correctness from first principles:
  - download skip/overwrite/fallback branches and output-path construction (network mocked);
  - date selection (`get_dates_to_process`, `get_existing_dates`), output-mode resolution
    (`determine_oscar_mode`), input normalization (`load_ds` renames / longitude wrapping /
    Kelvin→Celsius+mask / wind daily-mean), and the `write_oscar` output round-trip.
- `regression/` — the refactor anchor (runs the full `compute_surface_currents` pipeline):
  - `test_compute_regression.py` (fast, `do_eq=False`, synthetic global-lat inputs) — runs in CI.
  - `test_full_day_golden.py` (`@slow`, `do_eq=True`, real day-1 inputs) — local/nightly.
- `fixtures/` — committed KB-scale numerical baselines (`*_fingerprint.json`).

## Regenerating the fingerprints

The fingerprint captures per-field summary stats (`nanmin/max/mean/std`, finite count) plus
sampled point values for `Ug`/`Uw`/`Ub` (real & imag). Regenerate **only** when a change is a
deliberate, reviewed numerical improvement, and commit the JSON diff:

```bash
python tests/generate_reference.py          # both fingerprints (golden needs day-1 data)
python tests/generate_reference.py --fast   # fast fingerprint only
```

## Notes

- `oscar.computation.constants` reads `io_config.yaml` at import and bakes the global grid
  (`y`, `um`, `vm`, `eta`, spacing) into module constants. Tests run against the default
  config (cmems, 0.125°) and monkeypatch only where needed. This config-at-import coupling is
  known testability debt — a motivated first refactor now that the net exists.
- Fingerprints are generated with the pinned numpy-1.x stack in this repo's `.venv`; the
  tolerances (`rtol=1e-6`, `atol=1e-9`) absorb platform rounding while catching real drift.
