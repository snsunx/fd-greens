import sys
sys.path.append('../../src/')
import numpy as np
from ansatze import *
from hamiltonians import *
from greens_function import *

ansatz = build_kosugi_h2o_ansatz(2)
print(ansatz)
geometry = 'H {} {} 0; O 0 0 0; H {} {} 0'.format(
    -0.96 * np.cos(52.25 / 180 * np.pi), -0.96 * np.sin(52.25 / 180 * np.pi),
    0.96 * np.cos(52.25 / 180 * np.pi), -0.96 * np.sin(52.25 / 180 * np.pi))
hamiltonian = MolecularHamiltonian(geometry, 'sto3g')
greens_function = GreensFunction(ansatz, hamiltonian)
greens_function.compute_ground_state()

