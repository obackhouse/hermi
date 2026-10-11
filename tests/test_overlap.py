"""Integration tests for the overlap operator."""

from __future__ import annotations

import numpy as np
import pytest

from hermi.operators import Overlap


@pytest.mark.parametrize("spherical, suffix", ((False, "cart"), (True, "sph")))
def test_overlap_against_pyscf(example_system, normalized_reference, spherical: bool, suffix: str) -> None:
    """Compare overlap integrals against PySCF for each example system."""
    actual = Overlap().collect(example_system.basis, spherical=spherical)
    reference = normalized_reference(example_system, f"int1e_ovlp_{suffix}", suffix)

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("spherical", (False, True))
def test_overlap_ao_slices_match_full_array(example_system, spherical: bool) -> None:
    """Compare sliced overlap integrals with slices of the full matrix."""
    operator = Overlap()
    full = operator.collect(example_system.basis, spherical=spherical)
    num_aos = full.shape[0]
    ao_slices = (slice(None, None, 2), slice(None, None, -2))

    actual = operator.collect(example_system.basis, ao_slices=ao_slices, spherical=spherical)
    indices = tuple(np.arange(*ao_slice.indices(num_aos)) for ao_slice in ao_slices)
    expected = full[np.ix_(*indices)]

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
