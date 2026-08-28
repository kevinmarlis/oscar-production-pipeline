import numpy as np

from oscar.computation import constants as C
from oscar.computation.physics import (
    coriolis_polynomial,
    raw_velocity,
    rescale_velocities,
)


def test_coriolis_polynomial_matches_module_constant():
    # constants.f is built from the same Taylor expansion of f = 2*Omega*sin(phi).
    poly = coriolis_polynomial(C.PHIm)
    np.testing.assert_allclose(poly, C.f, rtol=1e-12, atol=0.0)


def test_coriolis_polynomial_is_odd_series():
    # Only odd powers of the argument survive in the sine expansion, so the even-power
    # coefficients (counting from the constant term at the end) are ~0.
    poly = coriolis_polynomial(C.PHIm)
    # poly has length 10 (degree 9); constant term poly[-1] must be 0 (sin(0)=0).
    assert poly[-1] == 0.0


def test_raw_velocity_zero_coriolis_gives_zero():
    F = np.array([1 + 2j, 3 + 4j, -5 + 0.5j])
    ff = np.zeros_like(F, dtype=float)
    out = raw_velocity(None, F, ff)
    np.testing.assert_array_equal(out, np.zeros_like(F))


def test_raw_velocity_formula():
    # U = F.imag/ff - 1j*F.real/ff  (i.e. U = F / (i f))
    F = np.array([1 + 2j, 3 + 4j])
    ff = np.array([2.0, 4.0])
    out = raw_velocity(None, F, ff)
    expected = F.imag / ff - 1j * (F.real / ff)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_rescale_velocities_scales_real_by_um_imag_by_vm():
    Ug = np.array([1 + 2j])
    Uw = np.array([3 + 4j])
    Ub = np.array([5 + 6j])
    um, vm = 0.4, 0.05

    Ugs, Uws, Ubs = rescale_velocities(Ug, Uw, Ub, um, vm)

    np.testing.assert_allclose(Ugs, um * Ug.real + 1j * vm * Ug.imag)
    np.testing.assert_allclose(Uws, um * Uw.real + 1j * vm * Uw.imag)
    np.testing.assert_allclose(Ubs, um * Ub.real + 1j * vm * Ub.imag)
