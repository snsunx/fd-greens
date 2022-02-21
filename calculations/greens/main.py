import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')

from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from eh_amplitudes_solver import EHAmplitudesSolver
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5

def main_gs():
    gs_solver = GroundStateSolver(h, q_instance=q_instance, method='exact', h5fname=h5fname)
    gs_solver.run()

def main_es():
    es_solver = EHStatesSolver(h, q_instance=q_instance, method='exact', h5fname=h5fname)
    es_solver.run()

def main_amp(**kwargs):
    amp_solver = EHAmplitudesSolver(h, q_instance=q_instance, h5fname=h5fname, suffix=suffix, anc=[0, 1])
    amp_solver.run(**kwargs)
    
def main_greens():
    greens_func = GreensFunction()
    greens_func.spectral_function(omegas, eta)
    greens_func.self_energy(omegas, eta)

if __name__ == '__main__': 
    h = get_lih_hamiltonian(3.0)
    q_instance = get_quantum_instance('sv')
    h5fname = 'lih_3A1'
    suffix = '_test'

    initialize_hdf5(h5fname)
    main_gs()
    main_es()
    main_amp()
    # main_greens()
