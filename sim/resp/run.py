import h5py
import numpy as np
import sys
sys.path.append('../..')
import warnings
warnings.filterwarnings('ignore')

from fd_greens.main import GroundStateSolver, ExcitedStatesSolver, ExcitedAmplitudesSolver, ResponseFunction
from fd_greens.main.ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from fd_greens.utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5
from fd_greens.main.params import HARTREE_TO_EV

def main_gs():
    gs_solver = GroundStateSolver(h, method='exact')
    gs_solver.run()

def main_es():
    es_solver = ExcitedStatesSolver(h)
    es_solver.run()

def main_amp(**kwargs):
    amp_solver = ExcitedAmplitudesSolver(h, q_instance=q_instance, h5fname=h5fname, method=method, suffix=suffix)
    amp_solver.run(**kwargs)

def main_resp():
    resp_func = ResponseFunction(h5fname=h5fname, suffix=suffix)
    resp_func.response_function(omegas, eta)

if __name__ == '__main__': 
    h = get_lih_hamiltonian(3.0)
    q_instance = get_quantum_instance('sv')
    method = 'exact'
    h5fname = 'lih'
    suffix = '_' + 'noisy'
    omegas = np.arange(-30, 30, 0.1)
    eta = 0.02 * HARTREE_TO_EV
    
    initialize_hdf5(calc='resp')
    main_gs()
    main_es()
    main_amp(method=method, build=True, execute=False, process=False)
    # main_resp()
