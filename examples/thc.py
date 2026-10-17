import time

import numpy as np

from hermi.algorithms.tensor_hypercontraction import (
    BlockTensorDecomposition,
    GlobalDecomposition,
    HybridGridPruner,
    RecursiveSpatialBisection,
    tensor_hypercontraction,
)
from hermi.systems.basis import Basis
from hermi.systems.grid import MolecularGrid
from hermi.systems.molecule import Molecule
from hermi.utils.linear_algebra import (
    pivoted_cholesky_pivots,
    greedy_pivoted_cholesky_pivots,
    randomized_pivoted_cholesky_pivots,
)


molecule = Molecule.from_list(
    ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
    [
        [ 2.64, 0.0, 0.0], [ 1.32,  2.2863, 0.0], [-1.32,  2.2863, 0.0],
        [-2.64, 0.0, 0.0], [-1.32, -2.2863, 0.0], [ 1.32, -2.2863, 0.0],
        [ 4.70, 0.0, 0.0], [ 2.35,  4.0703, 0.0], [-2.35,  4.0703, 0.0],
        [-4.70, 0.0, 0.0], [-2.35, -4.0703, 0.0], [ 2.35, -4.0703, 0.0],
    ],
    units="bohr",
)
ao_basis = Basis.from_molecule(molecule, "def2-svp")
aux_basis = Basis.from_molecule(molecule, "def2-universal-jfit")

start_time = time.time()

candidate_grid = MolecularGrid.from_becke(
    molecule,
    radial_order=16,
    angular_order=4,
    becke_iterations=3,
)
fit_grid = MolecularGrid.from_becke(
    molecule,
    radial_order=40,
    angular_order=10,
    becke_iterations=3,
)

sorter = RecursiveSpatialBisection(leaf_size=512)
candidate_pruner = BlockTensorDecomposition(
    block_size=256,
    cholesky_tolerance=1e-11,
    sorter=sorter,
    pivot_function=pivoted_cholesky_pivots,
)
global_pruner = GlobalDecomposition(
    cholesky_tolerance=1e-10,
    sorter=sorter,
    pivot_function=randomized_pivoted_cholesky_pivots,
    pivot_function_kwargs=dict(
        block_size=64,
        rng=np.random.default_rng(0),
    ),
)
pruner = HybridGridPruner(candidate_pruner=candidate_pruner, global_pruner=global_pruner)

collocation, kernel = tensor_hypercontraction(
    ao_basis,
    aux_basis,
    candidate_grid,
    fit_grid=fit_grid,
    grid_pruner=pruner,
)

print(f"Candidate grid: {len(candidate_grid)} points")
print(f"Fit grid:       {len(fit_grid)} points")
print(f"THC rank:       {collocation.shape[0]}")
print(f"Time elapsed:   {time.time() - start_time:.2f} seconds")
