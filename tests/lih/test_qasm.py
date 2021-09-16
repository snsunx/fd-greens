"""Calculates the spectral function on the active-space LiH Hamilonian.

This script was written before defining the classmethods to simplify things. 
For implementation using classmethods see test_qasm_new.py."""

import sys
sys.path.append('../../src/')
import numpy as np
from qiskit import *
from ansatze import *
from hamiltonians import MolecularHamiltonian
from qiskit.utils import QuantumInstance
from greens_function import GreensFunction
from tools import get_quantum_instance
from constants import HARTREE_TO_EV

# User-defined parameters.
bond_length = 1.6
cache_read = True
cache_write = True

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
greens_function_sv = GreensFunction(
    ansatz.copy(), hamiltonian, q_instance=q_instance_sv)
greens_function_sv.compute_ground_state()
greens_function_sv.compute_eh_states()
greens_function_sv.compute_diagonal_amplitudes(
    cache_read=cache_read, cache_write=cache_write)
greens_function_sv.compute_off_diagonal_amplitudes(
    cache_read=cache_read, cache_write=cache_write)
print(greens_function_sv.states_str_arr)

scaling_factor_h, constant_shift_h = \
    GreensFunction.get_hamiltonian_shift_parameters(
        hamiltonian, states_str='h')
scaling_factor_e, constant_shift_e = \
    GreensFunction.get_hamiltonian_shift_parameters(
        hamiltonian, states_str='e')

# QASM simulator calculation of h states
greens_function_h = GreensFunction(
    greens_function_sv.ansatz.copy(), hamiltonian, 
    q_instance=q_instance_noisy, 
    scaling=scaling_factor_h,
    shift=constant_shift_h,
    recompiled=True)
greens_function_h.states_str = 'h'
greens_function_h.states_str_arr = greens_function_sv.states_str_arr
greens_function_h.eigenstates_h = greens_function_sv.eigenstates_h
greens_function_h.compute_diagonal_amplitudes(
    cache_read=cache_read, cache_write=cache_write)
greens_function_h.compute_off_diagonal_amplitudes(
    cache_read=cache_read, cache_write=cache_write)

# QASM simulator calculation of e states
greens_function_e = GreensFunction(
    greens_function_sv.ansatz.copy(), hamiltonian, 
    q_instance=q_instance_noisy, 
    scaling=scaling_factor_e, 
    shift=constant_shift_e,
    recompiled=True)
greens_function_e.states_str = 'e'
greens_function_e.states_str_arr = greens_function_sv.states_str_arr
greens_function_h.eigenstates_e = greens_function_sv.eigenstates_e
greens_function_e.compute_diagonal_amplitudes(
    cache_read=cache_read, cache_write=cache_write)
greens_function_e.compute_off_diagonal_amplitudes(
    cache_read=cache_read, cache_write=cache_write)

# Combining h states and e states results
greens_function_final = GreensFunction(ansatz.copy(), hamiltonian)
greens_function_final.energy_gs = greens_function_sv.energy_gs
greens_function_final.eigenenergies_e = greens_function_sv.eigenenergies_e
greens_function_final.eigenenergies_h = greens_function_sv.eigenenergies_h

greens_function_final.B_e += greens_function_e.B_e
greens_function_final.D_ep += greens_function_e.D_ep
greens_function_final.D_em += greens_function_e.D_em

greens_function_final.B_h += greens_function_h.B_h
greens_function_final.D_hp += greens_function_h.D_hp
greens_function_final.D_hm += greens_function_h.D_hm

omegas = np.arange(-30, 34, 0.1)
As = []
for omega in omegas:
    A = greens_function_final.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    As.append(A)
np.savetxt('A_noisy.dat', np.vstack((omegas, As)).T)
