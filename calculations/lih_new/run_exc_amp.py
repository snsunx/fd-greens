import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from ground_state_solvers import GroundStateSolver
from number_states_solvers import ExcitedStatesSolver
from amplitudes_solvers import ExcitedAmplitudesSolver
from ansatze import build_ansatz_gs
from helpers import get_lih_hamiltonian, get_quantum_instance, save_exc_data

h = get_lih_hamiltonian(3.0)
q_inst_type = 'qasm'
q_instance = get_quantum_instance(q_inst_type)

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance)
gs_solver.run(method='exact')

es_solver = ExcitedStatesSolver(h)
es_solver.run(method='exact')

amp_solver = ExcitedAmplitudesSolver(h, gs_solver, es_solver)
amp_solver.run(method='exact')

save_exc_data(gs_solver, es_solver, amp_solver)
