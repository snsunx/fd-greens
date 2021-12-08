import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', message='')

import numpy as np
from ground_state_solvers import GroundStateSolver
from number_states_solvers import ExcitedStatesSolver
from amplitudes_solvers import ExcitedAmplitudesSolver
from response_function import ResponseFunction
from ansatze import build_ansatz_gs
from helpers import get_lih_hamiltonian, get_quantum_instance
from params import HARTREE_TO_EV

h = get_lih_hamiltonian(3.0)
q_inst_type = 'qasm'
q_instance = get_quantum_instance(q_inst_type)

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance)
gs_solver.run(method='exact')

exc_solver = ExcitedStatesSolver(h)
exc_solver.run(method='exact')

amp_solver = ExcitedAmplitudesSolver(h, gs_solver, exc_solver)
amp_solver.run(method='exact')

resp_func = ResponseFunction(gs_solver, exc_solver, amp_solver)
omegas = np.arange(-30, 30, 0.1)
chi00s = []
chi11s = []
sigmas = []

for omega in omegas:
    chi00 = resp_func.response_function(omega + 0.01j * HARTREE_TO_EV, 0, 0)
    chi00s.append(chi00)

    chi11 = resp_func.response_function(omega + 0.01j * HARTREE_TO_EV, 1, 1)
    chi11s.append(chi11)

    sigma = resp_func.cross_section(omega + 0.01j * HARTREE_TO_EV)
    sigmas.append(sigma)

chi00s = np.array(chi00s)
chi11s = np.array(chi11s)
sigmas = np.array(sigmas)

np.savetxt(f'data/chi00_{q_inst_type}.dat', np.vstack((omegas, chi00s.real, chi00s.imag)).T)
np.savetxt(f'data/chi11_{q_inst_type}.dat', np.vstack((omegas, chi11s.real, chi11s.imag)).T)
np.savetxt(f'data/sigma_{q_inst_type}.dat', np.vstack((omegas, sigmas.real, sigmas.imag)).T)
