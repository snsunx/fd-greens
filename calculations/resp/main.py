import h5py
import numpy as np
import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')

from ground_state_solvers import GroundStateSolver
from number_states_solvers import ExcitedStatesSolver
from excited_amplitudes_solver import ExcitedAmplitudesSolver
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5
from params import HARTREE_TO_EV
from response_function import ResponseFunction

def main_gs():
    gs_solver = GroundStateSolver(h)
    gs_solver.run()

def main_es():
    es_solver = ExcitedStatesSolver(h)
    es_solver.run()

def main_amp(**kwargs):
    amp_solver = ExcitedAmplitudesSolver(h, q_instance=q_instance)
    amp_solver.run(**kwargs)

def main_resp():
    resp_func = ResponseFunction()
    resp_func.response_function(omegas, eta)

if __name__ == '__main__': 
    h = get_lih_hamiltonian(3.0)
    q_instance = get_quantum_instance('qasm')
    omegas = np.arange(-30, 30, 0.1)
    eta = 0.02 * HARTREE_TO_EV
    
    initialize_hdf5(calc='resp')
    main_gs()
    main_es()
    main_amp(method='tomo')
    main_resp()
