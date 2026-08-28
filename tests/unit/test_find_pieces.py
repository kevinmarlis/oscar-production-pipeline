"""Tests for equatorial_treatment.find_pieces.

find_pieces splits a 1-D array into contiguous non-NaN "pieces", tolerating *isolated*
single NaNs (missing data) but splitting on runs of >=2 NaNs (interpreted as land/islands).
Pieces spanning fewer than 3 grid points (SUP - INF <= 1) are rejected. Behaviour below was
characterised against the trusted implementation.
"""

import numpy as np

from oscar.computation.equatorial_treatment import find_pieces

NAN = np.nan


def _run(seq):
    inf, sup = find_pieces(np.array(seq, dtype=float))
    return np.asarray(inf, dtype=int).tolist(), np.asarray(sup, dtype=int).tolist()


def test_all_valid_single_piece():
    assert _run([1, 2, 3, 4, 5]) == ([0], [4])


def test_isolated_nan_kept_as_single_piece():
    # One interior NaN is treated as missing data, not a boundary.
    assert _run([1, 2, NAN, 4, 5]) == ([0], [4])


def test_isolated_nan_in_long_run():
    assert _run([1, 2, 3, NAN, 5, 6, 7]) == ([0], [6])


def test_island_of_two_nans_splits_into_two_pieces():
    # Two long segments separated by a 2-NaN island -> two pieces.
    assert _run([1, 2, 3, NAN, NAN, 7, 8, 9]) == ([0, 5], [2, 7])


def test_short_pieces_rejected():
    # Two-NaN island leaves single-point stubs on each side; both are too short.
    assert _run([1, 2, NAN, NAN, 5, 6]) == ([], [])


def test_two_nan_gap_between_singletons_rejected():
    assert _run([1, NAN, NAN, 4]) == ([], [])


def test_nan_endpoints_trimmed():
    assert _run([NAN, 1, 2, 3, NAN]) == ([1], [3])


def test_all_nan_returns_empty():
    inf, sup = find_pieces(np.array([NAN, NAN, NAN], dtype=float))
    assert list(inf) == [] and list(sup) == []
