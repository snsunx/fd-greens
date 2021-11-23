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
save_params = False
load_params = False 
cache_read = False
cache_write = False

ansatz = build_2q_ansatz()
hamiltonians = []
radii = np.arange(1, 10, 0.2)
for r in radii:
    hamiltonian = get_lih_hamiltonian(r)
    hamiltonians.append(hamiltonian)
ccx_data = get_berkeley_ccx_data()

methods = {'gs': 'vqe', 'eh': 'ssvqe', 'amp': 'tomography'}
q_instance_type = 'sv'
q_instances = get_quantum_instance(q_instance_type)

E1_list = []
E2_list = []
for r, h in zip(radii, hamiltonians):
    print('%' * 20, 'r =', r, '%' * 20)
    gf = GreensFunctionRestricted(ansatz.copy(), h,
                                  methods=methods,
                                  q_instances=q_instances, 
                                  ccx_data=ccx_data, add_barriers=False,
                                  transpiled=True, recompiled=False,
                                  spin='down', push=True)
    gf.run(save_params=save_params, load_params=load_params, 
           cache_read=cache_read, cache_write=cache_write)
    E1, E2 = gf.get_correlation_energy()
    E1_list.append(E1)
    E2_list.append(E2)

E1_list = np.array(E1_list).real
E2_list = np.array(E2_list).real

fname = f'data/E1_red_{q_instance_type}1.dat'
np.savetxt(fname, np.vstack((radii, E1_list)).T)

fname = f'data/E2_red_{q_instance_type}1.dat'
np.savetxt(fname, np.vstack((radii, E2_list)).T)

"""
omegas = np.arange(-30, 30, 0.1)
A_list = []
for omega in omegas:
    A = gf.get_spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)
if methods['amp'] == 'energy':
    fname = f'data/A_red_{q_instance_type}e.dat'
else:
    fname = f'data/A_red_{q_instance_type}.dat'
np.savetxt(fname, np.vstack((omegas, A_list)).T)
"""
