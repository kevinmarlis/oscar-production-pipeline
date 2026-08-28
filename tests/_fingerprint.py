"""Compact numerical fingerprints for the regression safety net.

Instead of committing ~100 MB of global current output, we capture per-field summary
statistics plus a fixed set of sampled point values. The fingerprint is small
(KB-scale), diff-friendly JSON, and sensitive enough to catch silent numerical drift
introduced by a refactor.

A "field" here is a complex current component (``Ug``/``Uw``/``Ub``); we fingerprint its
real and imaginary parts separately.
"""

import json
import math

import numpy as np


def _sample_indices(shape, n_per_axis=4):
    """Deterministic (time, lat, lon) index tuples spread across the grid.

    Uses ``linspace`` endpoints so the same shape always yields the same samples,
    independent of platform or RNG.
    """
    nt, ny, nx = shape
    lats = np.unique(np.linspace(0, ny - 1, n_per_axis, dtype=int))
    lons = np.unique(np.linspace(0, nx - 1, n_per_axis, dtype=int))
    idx = []
    for t in range(nt):
        for j in lats:
            for i in lons:
                idx.append((int(t), int(j), int(i)))
    return idx


def _component_stats(values):
    """nan-aware summary stats for one real-valued array."""
    finite = np.isfinite(values)
    if finite.any():
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        vmean = float(np.nanmean(values))
        vstd = float(np.nanstd(values))
    else:
        vmin = vmax = vmean = vstd = math.nan
    return {
        "nanmin": vmin,
        "nanmax": vmax,
        "nanmean": vmean,
        "nanstd": vstd,
        "finite_count": int(finite.sum()),
    }


def fingerprint(components):
    """Build a fingerprint dict from ``{name: complex_ndarray}``.

    Each component contributes a ``<name>.real`` and ``<name>.imag`` entry plus sampled
    point values (NaNs serialised as ``null``).
    """
    fp = {"shape": None, "sample_indices": None, "fields": {}}
    ref_shape = None
    for name, arr in components.items():
        arr = np.asarray(arr)
        if ref_shape is None:
            ref_shape = arr.shape
            fp["shape"] = list(ref_shape)
            fp["sample_indices"] = [list(t) for t in _sample_indices(ref_shape)]
        elif arr.shape != ref_shape:
            raise ValueError(f"component {name!r} shape {arr.shape} != {ref_shape}")

        idx = [tuple(t) for t in fp["sample_indices"]]
        for part_name, part in (("real", arr.real), ("imag", arr.imag)):
            stats = _component_stats(part)
            samples = [None if not np.isfinite(part[t]) else float(part[t]) for t in idx]
            stats["samples"] = samples
            fp["fields"][f"{name}.{part_name}"] = stats
    return fp


def save(fp, path):
    with open(path, "w") as f:
        json.dump(fp, f, indent=2, sort_keys=True)
        f.write("\n")


def load(path):
    with open(path) as f:
        return json.load(f)


def _close(a, b, rtol, atol):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


def format_mismatch(problems, regenerate_hint, max_shown=8):
    """Turn a list of mismatch strings into a plain-language failure message.

    Written for non-software maintainers: it explains what a fingerprint mismatch *means*
    and the two things it could be (accidental drift vs a deliberate, reviewed change), and
    caps the raw diff so the output stays readable instead of scrolling off-screen.
    """
    shown = problems[:max_shown]
    hidden = len(problems) - len(shown)
    lines = [
        f"The computed currents no longer match the saved baseline "
        f"({len(problems)} value(s) changed).",
        "",
        "This means the numbers coming out of the pipeline are different from the",
        "reference that was committed. Either:",
        "  1. A change accidentally altered the science -> the currents drifted, "
        "investigate the change; or",
        "  2. The change is a deliberate, reviewed improvement -> update the baseline:",
        f"       {regenerate_hint}",
        "     then commit the updated fixtures/*.json.",
        "",
        "What changed (sampled):",
    ]
    lines += [f"  - {p}" for p in shown]
    if hidden > 0:
        lines.append(f"  ... and {hidden} more (see fixtures/*.json for the full baseline)")
    return "\n".join(lines)


def compare(reference, candidate, rtol=1e-6, atol=1e-9):
    """Return a list of human-readable mismatch strings (empty == match)."""
    problems = []
    if reference.get("shape") != candidate.get("shape"):
        problems.append(f"shape {candidate.get('shape')} != reference {reference.get('shape')}")
    if reference.get("sample_indices") != candidate.get("sample_indices"):
        problems.append("sample_indices differ")

    ref_fields = reference.get("fields", {})
    cand_fields = candidate.get("fields", {})
    if set(ref_fields) != set(cand_fields):
        problems.append(
            f"field set differs: reference={sorted(ref_fields)} candidate={sorted(cand_fields)}"
        )

    for name in sorted(set(ref_fields) & set(cand_fields)):
        r = ref_fields[name]
        c = cand_fields[name]
        for stat in ("nanmin", "nanmax", "nanmean", "nanstd"):
            if not _close(r.get(stat), c.get(stat), rtol, atol):
                problems.append(f"{name}.{stat}: {c.get(stat)!r} != reference {r.get(stat)!r}")
        if r.get("finite_count") != c.get("finite_count"):
            problems.append(
                f"{name}.finite_count: {c.get('finite_count')} != reference {r.get('finite_count')}"
            )
        r_samp = r.get("samples", [])
        c_samp = c.get("samples", [])
        if len(r_samp) != len(c_samp):
            problems.append(f"{name}.samples length differs")
        else:
            for k, (rv, cv) in enumerate(zip(r_samp, c_samp)):
                if not _close(rv, cv, rtol, atol):
                    problems.append(f"{name}.samples[{k}]: {cv!r} != reference {rv!r}")
    return problems
