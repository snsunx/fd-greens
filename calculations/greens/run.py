import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from eh_amplitudes_solver import EHAmplitudesSolver
from greens_function import GreensFunction
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5
from params import HARTREE_TO_EV

def main_gs():
    gs_solver = GroundStateSolver(h, q_instance=q_instance, method='exact', h5fname=h5fname)
    gs_solver.run()

def main_es():
    es_solver = EHStatesSolver(h, q_instance=q_instance, method='exact', h5fname=h5fname, spin=spin)
    es_solver.run()

def main_amp(**kwargs):
    amp_solver = EHAmplitudesSolver(h, q_instance=q_instance, h5fname=h5fname, method=method, 
                                    suffix=suffix, anc=[0, 1], spin=spin)
    amp_solver.run(**kwargs)
    
def main_greens():
    greens_func = GreensFunction(h5fname=h5fname, suffix=suffix)
    greens_func.spectral_function(omegas, eta)
    greens_func.self_energy(omegas, eta)

if __name__ == '__main__': 
    h = get_lih_hamiltonian(3.0)
    q_instance = get_quantum_instance('sv')
    method = 'exact'
    h5fname = 'lih_3A'
    spin = 'd'
    suffix = '_' + spin
    omegas = np.arange(-20, 20, 0.01)
    eta = 0.02 * HARTREE_TO_EV

    initialize_hdf5(h5fname)
    main_gs()
    main_es()
    main_amp()
    main_greens()
