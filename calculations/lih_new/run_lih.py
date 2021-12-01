import sys
import warnings
sys.path.append('../../src/')
#warnings.filterwarnings('ignore', message='')

import numpy as np
from helpers import get_lih_hamiltonian, get_quantum_instance
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from vqe import GroundStateSolver
from number_state_solvers import EHStatesSolver
from amplitudes_solver import AmplitudesSolver
from greens_function import GreensFunction
from constants import HARTREE_TO_EV

h = get_lih_hamiltonian(3.0)
q_instance_type = 'qasm'
q_instance = get_quantum_instance(q_instance_type)

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance)
gs_solver.run(method='exact')

eh_solver = EHStatesSolver(h, ansatz_func_e=build_ansatz_e, ansatz_func_h=build_ansatz_h, q_instance=q_instance)
eh_solver.run(method='exact')

amp_solver = AmplitudesSolver(h, gs_solver, eh_solver, q_instance=q_instance)
amp_solver.run(method='energy')

gf = GreensFunction(gs_solver, eh_solver, amp_solver)
omegas = np.arange(-30, 30, 0.1)
A_list = []
TrS_list = []

for omega in omegas:
    A = gf.spectral_function(omega + 0.02j * HARTREE_TO_EV)
    A_list.append(A)

    Sigma = gf.self_energy(omega + 0.02j * HARTREE_TO_EV)
    TrS_list.append(np.trace(Sigma))

fname = f'data/A_red_{q_instance_type}.dat'
np.savetxt(fname, np.vstack((omegas, A_list)).T)

fname = f'data/TrS_red_{q_instance_type}.dat'
np.savetxt(fname, np.vstack((omegas, np.real(TrS_list), np.imag(TrS_list))).T)
