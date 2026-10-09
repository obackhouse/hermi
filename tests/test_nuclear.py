"""Integration tests for the nuclear attraction operator."""

from __future__ import annotations

import numpy as np
import pytest

from hermi.operators import Nuclear


@pytest.mark.parametrize("spherical, suffix", ((False, "cart"), (True, "sph")))
def test_nuclear_against_pyscf(example_system, normalized_reference, spherical: bool, suffix: str) -> None:
    """Compare nuclear attraction integrals against PySCF for each example system."""
    actual = Nuclear(example_system.molecule).collect(example_system.basis, spherical=spherical)
    reference = normalized_reference(example_system, f"int1e_nuc_{suffix}", suffix)

    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-8)
