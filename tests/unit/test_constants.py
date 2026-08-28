import numpy as np
import pytest

from oscar.computation import constants as C
from oscar.computation.physics import thermal_expansion_coefficient


# Reference value locked from the trusted numpy-1.x code at the model reference
# state (T=26 degC, S=35 psu). This anchors Gill's density polynomial.
CHI_REF_26_35 = 0.000304615446220295


def test_thermal_expansion_reference_value():
    chi = thermal_expansion_coefficient(26.0, 35.0)
    assert float(chi) == pytest.approx(CHI_REF_26_35, rel=1e-12)


def test_constants_chim_matches_reference_state():
    # CHIm in constants.py is computed at (Tm=26, Sm=35); it must equal the direct call.
    assert float(C.CHIm) == pytest.approx(CHI_REF_26_35, rel=1e-12)
    assert (C.Tm, C.Sm) == (26, 35)


def test_thermal_expansion_positive_over_ocean_range():
    for T in (0.0, 10.0, 20.0, 30.0):
        assert float(thermal_expansion_coefficient(T, 35.0)) > 0.0


def test_thermal_expansion_monotonic_increasing_with_temperature():
    temps = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    chi = np.array([float(thermal_expansion_coefficient(T, 35.0)) for T in temps])
    assert np.all(np.diff(chi) > 0.0)
