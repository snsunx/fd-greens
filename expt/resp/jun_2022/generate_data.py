import sys
sys.path.append('../../..')

import numpy as np
import cirq
from fd_greens import ResponseFunction, get_lih_hamiltonian

from fd_greens.cirq_ver.parameters import HARTREE_TO_EV

def generate_data(fname, spin):
	resp = ResponseFunction(hamiltonian, fname=fname, method='tomo')
	resp.response_function(omegas, eta)

if __name__ == '__main__':
	hamiltonian = get_lih_hamiltonian(3.0)
	omegas = np.arange(-20, 20, 0.01)
	eta = 0.02 * HARTREE_TO_EV
	for fname in ['lih_resp_sim', 'lih_3A_expt0', 'lih_3A_expt1', 'lih_3A_expt2', 'lih_3A_expt3']:
		generate_data(fname, spin='d')


