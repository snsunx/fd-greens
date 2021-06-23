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

ansatz = build_ne2_ansatz(4)
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, 1.6)]], 'sto3g', 
    occupied_inds=[0], active_inds=[1, 2])
q_instance_sv = QuantumInstance(Aer.get_backend('statevector_simulator'))
q_instance_qasm = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=8192)
q_instance_noisy = get_quantum_instance(
    Aer.get_backend('qasm_simulator'), shots=8192, noise_model_name='ibmq_jakarta')

# Statevector simulator calculation
greens_function_sv = GreensFunction(ansatz.copy(), hamiltonian, q_instance=q_instance_sv)
greens_function_sv.compute_ground_state()
greens_function_sv.compute_eh_states()
greens_function_sv.compute_diagonal_amplitudes()
greens_function_sv.compute_off_diagonal_amplitudes()
print(greens_function_sv.states_arr)

scaling_factor_h, constant_shift_h = \
    GreensFunction.get_hamiltonian_shift_parameters(hamiltonian, states='h')
scaling_factor_e, constant_shift_e = \
    GreensFunction.get_hamiltonian_shift_parameters(hamiltonian, states='e')

# QASM simulator calculation of h states
greens_function_h = GreensFunction(
    greens_function_sv.ansatz.copy(), hamiltonian, 
    q_instance=q_instance_noisy, 
    scaling_factor=scaling_factor_h,
    constant_shift=constant_shift_h,
    recompiled=True)
greens_function_h.states = 'h'
greens_function_h.states_arr = greens_function_sv.states_arr
greens_function_h.eigenstates_h = greens_function_sv.eigenstates_h
greens_function_h.compute_diagonal_amplitudes()
greens_function_h.compute_off_diagonal_amplitudes()

# QASM simulator calculation of e states
greens_function_e = GreensFunction(
    greens_function_sv.ansatz.copy(), hamiltonian, 
    q_instance=q_instance_noisy, 
    scaling_factor=scaling_factor_e, 
    constant_shift=constant_shift_e,
    recompiled=True)
greens_function_e.states = 'e'
greens_function_e.states_arr = greens_function_sv.states_arr
greens_function_h.eigenstates_e = greens_function_sv.eigenstates_e
greens_function_e.compute_diagonal_amplitudes()
greens_function_e.compute_off_diagonal_amplitudes()

# Combining h states and e states results
greens_function_final = GreensFunction(ansatz.copy(), hamiltonian)
greens_function_final.energy_gs = greens_function_sv.energy_gs
greens_function_final.energies_e = greens_function_sv.energies_e
greens_function_final.energies_h = greens_function_sv.energies_h

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
