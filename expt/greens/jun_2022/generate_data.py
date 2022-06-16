import sys
sys.path.append('../../..')

import numpy as np
import cirq
from fd_greens import (
	GroundStateSolver, 
	EHStatesSolver, 
	EHAmplitudesSolver, 
	GreensFunction, 
	get_lih_hamiltonian, 
	initialize_hdf5)

from fd_greens.cirq_ver.parameters import HARTREE_TO_EV

def generate_data(fname, spin):
	greens = GreensFunction(hamiltonian, fname=fname, method='tomo', spin=spin)
	greens.spectral_function(omegas, eta)
	greens.self_energy(omegas, eta)
	

if __name__ == '__main__':
	hamiltonian = get_lih_hamiltonian(3.0)
	omegas = np.arange(-20, 20, 0.01)
	eta = 0.02 * HARTREE_TO_EV
	for fname in ['lih_3A_sim', 'lih_3A_expt0', 'lih_3A_expt1', 'lih_3A_expt2', 'lih_3A_expt3']:
		generate_data(fname, spin='d')


