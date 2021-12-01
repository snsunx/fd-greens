import sys
import warnings
sys.path.append('../../src/')
#warnings.filterwarnings('ignore', message='')

from helpers import get_lih_hamiltonian, get_quantum_instance
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from vqe import GroundStateSolver
from number_state_solvers import EHStatesSolver
from amplitudes_solver import AmplitudesSolver

h = get_lih_hamiltonian(3.0)
q_instance = get_quantum_instance('sv')


gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance)
gs_solver.run(method='exact')

eh_solver = EHStatesSolver(h, ansatz_func_e=build_ansatz_e, ansatz_func_h=build_ansatz_h, q_instance=q_instance)
eh_solver.run(method='exact')

amp_solver = AmplitudesSolver(h, gs_solver, eh_solver, method='exact', q_instance=q_instance)
amp_solver.run()

