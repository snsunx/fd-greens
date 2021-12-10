import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from amplitudes_solvers import EHAmplitudesSolver
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from helpers import get_lih_hamiltonian, get_quantum_instance, save_eh_data
from utils import get_statevector

h = get_lih_hamiltonian(3.0)
q_instance = get_quantum_instance('qasm')

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance)
gs_solver.run(method='vqe')
#sv = get_statevector(gs_solver.ansatz)

es_solver = EHStatesSolver(h, ansatz_func_e=build_ansatz_e, ansatz_func_h=build_ansatz_h, q_instance=q_instance)
es_solver.run(method='vqe')

amp_solver = EHAmplitudesSolver(h, gs_solver, es_solver, q_instance=q_instance)
amp_solver.run(method='tomo')

save_eh_data(gs_solver, es_solver, amp_solver, fname='lih_tomo')
