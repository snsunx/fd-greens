"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from itertools import product

import numpy as np

# from .utilities import reverse_qubit_order

HARTREE_TO_EV = 27.211386245988

REVERSE_QUBIT_ORDER = True

# TODO: Write this into a class and include the tapered state in the class.
method_indices_pairs = {
    'u': [("cnot", (2, 0)), ("cnot", (3, 1)), ("taper", (0, 1))],
    'd': [("cnot", (2, 0)), ("cnot", (3, 1)), ("swap", (2, 3)), ("taper", (0, 1))]
    }

# TODO: This should be moved to another place.
# Basis matrix for tomography
basis_matrix = []
if REVERSE_QUBIT_ORDER:
    bases = [(f'{x[0]}{x[3]}', f'{x[1]}{x[2]}') for x in product('xyz', 'xyz', '01', '01')]
else:
    bases = [(f'{x[0]}{x[2]}', f'{x[1]}{x[3]}') for x in product('xyz', 'xyz', '01', '01')]

states = {
    "x0": np.array([1.0, 1.0]) / np.sqrt(2),
    "x1": np.array([1.0, -1.0]) / np.sqrt(2),
    "y0": np.array([1.0, 1.0j]) / np.sqrt(2),
    "y1": np.array([1.0, -1.0j]) / np.sqrt(2),
    "z0": np.array([1.0, 0.0]),
    "z1": np.array([0.0, 1.0]),
}

for basis in bases:
    if REVERSE_QUBIT_ORDER:
        basis_state = np.kron(states[basis[1]], states[basis[0]])
    else:
        basis_state = np.kron(states[basis[0]], states[basis[1]])
    basis_vectorized = np.outer(basis_state, basis_state.conj()).reshape(-1)
    basis_matrix.append(basis_vectorized)

basis_matrix = np.array(basis_matrix)
