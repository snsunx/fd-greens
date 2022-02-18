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

def main_gs_es():
    gs_solver = GroundStateSolver(h)
    gs_solver.run()

    es_solver = ExcitedStatesSolver(h)
    es_solver.run()

def main_amp(method='exact'):
    amp_solver = ExcitedAmplitudesSolver(h)
    amp_solver.build_diagonal()
    amp_solver.run_diagonal()
    amp_solver.process_diagonal()
    amp_solver.build_off_diagonal()
    amp_solver.run_off_diagonal()
    amp_solver.process_off_diagonal()

def main_resp():
    resp_func = ResponseFunction()
    resp_func.response_function(omegas, eta)

if __name__ == '__main__': 
    h = get_lih_hamiltonian(3.0)
    omegas = np.arange(-30, 30, 0.1)
    eta = 0.02 * HARTREE_TO_EV
    
    initialize_hdf5(calc='resp')
    #main_gs_es()
    #main_amp()
    main_resp()
