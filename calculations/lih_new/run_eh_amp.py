import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from amplitudes_solvers import EHAmplitudesSolver
from greens_function import GreensFunction

from helpers import get_lih_hamiltonian, get_quantum_instance, save_solvers_data
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from params import HARTREE_TO_EV


h = get_lih_hamiltonian(3.0)
q_inst_type = 'sv'
q_instance = get_quantum_instance(q_inst_type)

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance)
gs_solver.run(method='exact')

es_solver = EHStatesSolver(h, ansatz_func_e=build_ansatz_e, ansatz_func_h=build_ansatz_h, q_instance=q_instance)
es_solver.run(method='exact')

amp_solver = EHAmplitudesSolver(h, gs_solver, es_solver, q_instance=q_instance)
amp_solver.run(method='exact')

save_solvers_data(gs_solver, es_solver, amp_solver)
