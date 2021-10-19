"""Calculates the spectral function on the active-space LiH Hamilonian 
using the classmethods."""

import sys
sys.path.append('../../src/')
import numpy as np
from qiskit import *
from qiskit.circuit import Barrier
from ansatze import *
from hamiltonians import MolecularHamiltonian
from qiskit.utils import QuantumInstance
from qiskit.extensions import CCXGate, UnitaryGate, SwapGate

from greens_function_restricted import GreensFunctionRestricted
from utils import get_quantum_instance
from constants import HARTREE_TO_EV

# User-defined parameters.
bond_length = 3.0
save_params = False 
load_params = True
cache_read = False
cache_write = False

#ccx_data = [(CCXGate(), [0, 1, 2])]
iX = np.array([[0, 1j], [1j, 0]])
ccx_data = [(SwapGate(), [1, 2]), 
            (Barrier(4), [0, 1, 2, 3]), 
            (UnitaryGate(iX).control(2), [0, 2, 1]), 
            (Barrier(4), [0, 1, 2, 3]),
            (SwapGate(), [1, 2])]

#ansatz = build_two_local_ansatz(2)
ansatz = build_2q_ansatz()
hamiltonian = MolecularHamiltonian(
    [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]], 'sto3g', 
    occ_inds=[0], act_inds=[1, 2])

q_instance_sv = QuantumInstance(Aer.get_backend('statevector_simulator'))
q_instance_qasm = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=20000)
q_instance_noisy = get_quantum_instance(
    Aer.get_backend('qasm_simulator'), shots=20000, noise_model_name='ibmq_jakarta')

# Statevector simulator calculation
print("========== Starts statevector simulation ==========")
gf_sv = GreensFunctionRestricted(ansatz.copy(), hamiltonian, 
								 q_instance=q_instance_noisy, 
    							 ccx_data=ccx_data, add_barriers=False,
                                 transpiled=True, recompiled=False,
                                 spin='down', push=True)
gf_sv.run(save_params=save_params, load_params=load_params, 
    	  cache_read=cache_read, cache_write=cache_write)

omegas = np.arange(-30, 30, 0.1)
A_list = []
for omega in omegas:
    A = gf_sv.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)
np.savetxt('A_red_qasm.dat', np.vstack((omegas, A_list)).T)
