"""Calculates the spectral function on the active-space LiH Hamilonian 
using the classmethods."""

import sys
sys.path.append('../../src/')
import numpy as np
from qiskit import *
from ansatze import build_ne2_ansatz
from hamiltonians import MolecularHamiltonian
from qiskit.utils import QuantumInstance
from greens_function import GreensFunction
from tools import get_quantum_instance
from constants import HARTREE_TO_EV

# User-defined parameters.
bond_length = 1.6
cache_read = True
cache_write = False

ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]], 'sto3g', 
    occupied_inds=[0], active_inds=[1, 2])

q_instance_sv = QuantumInstance(Aer.get_backend('statevector_simulator'))
q_instance_qasm = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8192)
q_instance_noisy = get_quantum_instance(
    Aer.get_backend('qasm_simulator'), shots=8192, 
    noise_model_name='ibmq_jakarta')

# Statevector simulator calculation
print("========== Starts statevector simulation ==========")
gf_sv = GreensFunction(ansatz.copy(), hamiltonian, q_instance=q_instance_sv)
gf_sv.run(cache_read=cache_read, cache_write=cache_write)
#print(gf_sv.states_arr)
exit()

# QASM simulator calculation of h states
print("============ Starts calculating h states ==========")
gf_h = GreensFunction.initialize_eh(gf_sv, 'h', q_instance=q_instance_sv)
gf_h.run(compute_energies=False, cache_read=cache_read, cache_write=cache_write)
print(gf_h.states_arr)

# QASM simulator calculation of e states
print("========== Starts calculating e states ==========")
gf_e = GreensFunction.initialize_eh(gf_sv, 'e', q_instance=q_instance_sv)
gf_e.run(compute_energies=False, cache_read=cache_read, cache_write=cache_write)
print(gf_e.states_arr)

# Combining h states and e states results
gf_final = GreensFunction.initialize_final(
    gf_sv, gf_e, gf_h, q_instance=q_instance_noisy)

omegas = np.arange(-30, 34, 0.1)
A_list = []
for omega in omegas:
    A = gf_final.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)
np.savetxt('A_noisy_cached2.dat', np.vstack((omegas, A_list)).T)
