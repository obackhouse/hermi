"""Hermite operator contraction tests."""

from __future__ import annotations

import numpy as np
import pytest

from hermi.integrals.contract import (
    contract_operator_10,
    contract_operator_11,
    contract_operator_12,
    contract_operator_20,
    contract_operator_22,
)
from hermi.operators.coulomb import Coulomb
from hermi.operators.operator import Operator
from hermi.operators.overlap import Overlap
from hermi.operators.point_charge import PointCharge


def _contract(
    signature: tuple[int, int],
    views,
    operator_matrix: np.ndarray,
    expansion,
    *,
    spherical: bool,
    batch_dim: bool,
) -> np.ndarray:
    """Call the contraction helper for an operator signature."""
    if signature == (1, 0):
        return contract_operator_10(
            operator_matrix, views[0], expansion[0], spherical=spherical, batch_dim=batch_dim
        )
    if signature == (1, 1):
        return contract_operator_11(
            operator_matrix, *views, *expansion, spherical=spherical, batch_dim=batch_dim
        )
    if signature == (1, 2):
        return contract_operator_12(
            operator_matrix, views[0], views[1], *expansion, spherical=spherical, batch_dim=batch_dim
        )
    if signature == (2, 0):
        return contract_operator_20(
            operator_matrix, views[0], expansion[0], spherical=spherical, batch_dim=batch_dim
        )
    if signature == (2, 2):
        return contract_operator_22(
            operator_matrix, *views, *expansion, spherical=spherical, batch_dim=batch_dim
        )
    raise AssertionError(f"unsupported test signature {signature}")


@pytest.mark.parametrize(
    ("operator", "signature"),
    (
        (PointCharge(np.asarray([0.1, -0.2, 0.3], dtype=np.float64)), (1, 0)),
        (Overlap(), (1, 1)),
        (Overlap(), (2, 0)),
        (Coulomb(), (1, 2)),
        (Coulomb(), (2, 2)),
    ),
)
@pytest.mark.parametrize("spherical", (False, True))
def test_contract_preserves_batch_dimension(
    basis_factory, operator: Operator, signature: tuple[int, int], spherical: bool
) -> None:
    """Each contraction helper preserves independent operator matrices along axis zero."""
    basis = basis_factory(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.8]],
        [0.7, 0.5],
        [0, 1],
    )
    shells = list(basis.iter_shell_groups())
    pair = basis[shells[0], shells[1]]
    views = {
        (1, 0): (shells[0], None),
        (1, 1): (shells[0], shells[1]),
        (1, 2): (shells[0], pair),
        (2, 0): (pair, None),
        (2, 2): (pair, pair),
    }[signature]
    operator_matrix = operator.get_operator_matrix(views)
    expansion = operator.get_expansion_matrices(views, spherical=spherical)
    scales = (1.5, -0.25)
    operator_batch = np.stack([scale * operator_matrix for scale in scales])

    reference = _contract(
        signature,
        views,
        operator_matrix,
        expansion,
        spherical=spherical,
        batch_dim=False,
    )
    actual = _contract(
        signature,
        views,
        operator_batch,
        expansion,
        spherical=spherical,
        batch_dim=True,
    )
    summed = _contract(
        signature,
        views,
        operator_batch,
        expansion,
        spherical=spherical,
        batch_dim=False,
    )

    assert actual.shape == (len(scales), *reference.shape)
    np.testing.assert_allclose(actual, np.stack([scale * reference for scale in scales]))
    np.testing.assert_allclose(summed, sum(scales) * reference)
