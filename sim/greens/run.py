import sys
sys.path.append('../../')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from fd_greens.main import GroundStateSolver, EHStatesSolver, EHAmplitudesSolver, GreensFunction
from fd_greens.main.ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from fd_greens.main.params import HARTREE_TO_EV
from fd_greens.utils import initialize_hdf5, get_lih_hamiltonian, get_quantum_instance

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
    h5fname = 'lih_3A1'
    spin = 'd'
    suffix = '_' + spin
    omegas = np.arange(-20, 20, 0.1)
    eta = 0.02 * HARTREE_TO_EV

    initialize_hdf5(h5fname)
    main_gs()
    main_es()
    main_amp()
    main_greens()
