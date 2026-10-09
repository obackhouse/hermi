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
