"""Composite operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hermi.operators.kinetic import Kinetic
from hermi.operators.nuclear import Nuclear
from hermi.operators.operator import OperatorSum

if TYPE_CHECKING:
    from dataclasses import InitVar

    from hermi.systems.molecule import Molecule


@dataclass(frozen=True)
class CoreHamiltonian(OperatorSum):
    r"""Core Hamiltonian operator.

    .. math::
        \hat{O} = -\frac{1}{2} \nabla^2 - \sum_C \frac{Z_C}{|r - R_C|}.
    """

    molecule: InitVar[Molecule] = field(repr=False, compare=False)

    def __init__(self, molecule: Molecule) -> None:
        kinetic = Kinetic()
        nuclear = Nuclear(molecule)
        super().__init__(operators=(kinetic, nuclear))
