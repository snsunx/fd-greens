"""Calculates the spectral function on the active-space LiH Hamilonian 
using the classmethods."""

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
q_instance_type = 'qasm'
ccx_data = get_berkeley_ccx_data()

# Statevector simulator calculation
print("========== Starts statevector simulation ==========")
gf = GreensFunctionRestricted(ansatz.copy(), hamiltonian, 
                              q_instance=get_quantum_instance(q_instance_type), 
    						  ccx_data=ccx_data, add_barriers=False,
                              transpiled=True, recompiled=False,
                              spin='down', push=True)
gf.run(save_params=save_params, load_params=load_params, 
    	  cache_read=cache_read, cache_write=cache_write)

omegas = np.arange(-30, 30, 0.1)
A_list = []
for omega in omegas:
    A = gf.compute_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)
np.savetxt(f'A_red_{q_instance_type}.dat', np.vstack((omegas, A_list)).T)
