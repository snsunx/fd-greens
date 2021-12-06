import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', message='')

from vqe import GroundStateSolver
from helpers import get_lih_hamiltonian, get_quantum_instance
from ansatze import build_ansatz_gs
from number_state_solvers import ExcitedStatesSolver
from response_function import ResponseFunction
from excited_amplitudes_solver import ExcitedAmplitudesSolver

h = get_lih_hamiltonian(3.0)
q_instance_type = 'qasm'
q_instance = get_quantum_instance(q_instance_type)

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance)
gs_solver.run(method='exact')

exc_solver = ExcitedStatesSolver(h)
exc_solver.run(method='exact')

amp_solver = ExcitedAmplitudesSolver(h, gs_solver, exc_solver)
amp_solver.run(method='exact')
