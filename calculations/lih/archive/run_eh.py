import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from amplitudes_solvers import EHAmplitudesSolver
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5

h = get_lih_hamiltonian(3.0)
q_instance = get_quantum_instance('qasm')
h5fname = 'lih_u'
initialize_hdf5(h5fname)

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance, h5fname=h5fname)
gs_solver.run(method='exact')

es_solver = EHStatesSolver(h, ansatz_func_e=build_ansatz_e, ansatz_func_h=build_ansatz_h, 
                           q_instance=q_instance, h5fname=h5fname)
es_solver.run(method='exact')

amp_solver = EHAmplitudesSolver(h, q_instance=q_instance, transpiled=False, swap_gates_pushed=False, h5fname=h5fname)
amp_solver.run(method='tomo')
