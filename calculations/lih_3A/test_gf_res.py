"""Calculates the spectral function on the active-space LiH Hamilonian."""

import sys
sys.path.append('../../src/')
import numpy as np
from qiskit import *
from ansatze import *

from greens_function_restricted import GreensFunctionRestricted
from constants import HARTREE_TO_EV

from helpers import get_berkeley_ccx_data, get_lih_hamiltonian, get_quantum_instance

# User-defined parameters
save_params = True
load_params = False
cache_read = False
cache_write = False

ansatz = build_2q_ansatz()
hamiltonian = get_lih_hamiltonian(3.0)
ccx_data = get_berkeley_ccx_data()
q_instance_types = ['qasm', 'sv', 'qasm']
q_instances = [get_quantum_instance(s) for s in q_instance_types]

# Statevector simulator calculation
print("========== Starts statevector simulation ==========")
gf = GreensFunctionRestricted(ansatz.copy(), hamiltonian, 
                              q_instances=q_instances, 
    						  ccx_data=ccx_data, add_barriers=False,
                              transpiled=True, recompiled=False,
                              spin='down', push=True)
gf.run(save_params=save_params, load_params=load_params, 
    	  cache_read=cache_read, cache_write=cache_write)

omegas = np.arange(-30, 30, 0.1)
A_list = []
for omega in omegas:
    A = gf.get_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)
np.savetxt(f'data/A_red_{"-".join(q_instance_types)}.dat', np.vstack((omegas, A_list)).T)
