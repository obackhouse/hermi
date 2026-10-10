# hermi

The `hermi` package implements various integrals over atom-centred Gaussian basis functions for use in quantum chemistry.
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
and [by Fermann and Valeev](https://arxiv.org/abs/2506.12501).
