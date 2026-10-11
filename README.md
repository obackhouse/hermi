# hermi

![Hermi logo](https://raw.githubusercontent.com/obackhouse/hermi/main/assets/hermi-banner.svg)

The `hermi` package implements integrals over atom-centred Gaussian basis functions for use in quantum chemistry.
It implements a generalised McMurchie--Davidson algorithm which targets matrix operations in order to effectively vectorise the workload via NumPy calls.
This permits efficient integral generation without any custom compiled code.

## Installation

The package can be installed from the PyPI repository:

```sh
pip install hermi
```

Alternatively, clone the repository and install via the source code:

```sh
git clone https://github.com/obackhouse/hermi.git
cd hermi && pip install .
```

## Usage

Helper classes for molecules and basis sets are implemented:

```python
from hermi import Molecule, Basis
molecule = Molecule.from_list(["Li", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.64]])
basis = Basis.from_molecule(molecule, "6-31g")
```

The automatic basis construction uses the `basis-set-exchange` library.
Operators can then be constructed and evaluated over the basis set:

```python
from hermi import Overlap
overlap = Overlap()
matrix = overlap.collect(basis)
```

Operators are also composable; along with lazy evaluation and caching,
this can afford optimisations when computing molecular integrals summed over a number of operators:

```python
from hermi import Kinetic, Nuclear
core_hamiltonian = Kinetic() + Nuclear(molecule)
matrix = core_hamiltonian.collect(basis)
```

## Development

We recommend `uv` for development, install development dependencies and run the test suite with:

```sh
uv sync --extra dev
uv run pytest
```

The operator integration tests compare Cartesian and spherical integrals against PySCF.

## References

This work is based on several papers:
first, the [original paper of McMurchie and Davidson](https://doi.org/10.1016/0021-9991(78)90092-X),
and more specifically,
the modern implementations detailed in papers [by Neese](https://doi.org/10.1002/jcc.26942)
and [by Asadchev and Valeev](https://arxiv.org/abs/2506.12501).
