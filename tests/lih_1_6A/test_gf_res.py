"""Calculates the spectral function on the active-space LiH Hamilonian 
using the classmethods."""

import sys
sys.path.append('../../src/')
import numpy as np
from qiskit import *
from ansatze import *
from hamiltonians import MolecularHamiltonian
from qiskit.utils import QuantumInstance
from greens_function_restricted import GreensFunctionRestricted
from utils import get_quantum_instance
from constants import HARTREE_TO_EV

# User-defined parameters.
bond_length = 1.6
save_params = False
load_params = False
cache_read = False
cache_write = False

ansatz = build_two_local_ansatz(2)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]], 'sto3g', 
    occ_inds=[0], act_inds=[1, 2])

q_instance_sv = QuantumInstance(Aer.get_backend('statevector_simulator'))
q_instance_qasm = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8192)
q_instance_noisy = get_quantum_instance(
    Aer.get_backend('qasm_simulator'), shots=8192, 
    noise_model_name='ibmq_jakarta')

# Statevector simulator calculation
print("========== Starts statevector simulation ==========")
gf_sv = GreensFunctionRestricted(ansatz.copy(), hamiltonian, q_instance=q_instance_noisy)
gf_sv.run(save_params=save_params, load_params=load_params, 
		  cache_read=cache_read, cache_write=cache_write)

omegas = np.arange(-30, 34, 0.1)
A_list = []
for omega in omegas:
    A = gf_sv.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)
np.savetxt('A_red_qasm.dat', np.vstack((omegas, A_list)).T)
