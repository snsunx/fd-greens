"""
========================================
Parameters (:mod:`fd_greens.parameters`)
========================================
"""

from itertools import product

import numpy as np

HARTREE_TO_EV = 27.211386245988

method_indices_pairs = {'u': [("cnot", (2, 0)), ("cnot", (3, 1)), ("taper", (0, 1))],
                        'd': [("cnot", (2, 0)), ("cnot", (3, 1)), ("swap", (2, 3)), ("taper", (0, 1))]}

# TODO: This should be moved to another place.
# Basis matrix for tomography
basis_matrix = []
bases = list(product("xyz", "xyz", "01", "01"))
states = {
    "x0": np.array([1.0, 1.0]) / np.sqrt(2),
    "x1": np.array([1.0, -1.0]) / np.sqrt(2),
    "y0": np.array([1.0, 1.0j]) / np.sqrt(2),
    "y1": np.array([1.0, -1.0j]) / np.sqrt(2),
    "z0": np.array([1.0, 0.0]),
    "z1": np.array([0.0, 1.0]),
}

for basis in bases:
    label0 = "".join([basis[0], basis[3]])
    label1 = "".join([basis[1], basis[2]])
    state0 = states[label0]
    state1 = states[label1]
    state = np.kron(state1, state0)
    rho_vec = np.outer(state, state.conj()).reshape(-1)
    basis_matrix.append(rho_vec)

basis_matrix = np.array(basis_matrix)
