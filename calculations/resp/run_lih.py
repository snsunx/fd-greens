import h5py
import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')

from ground_state_solvers import GroundStateSolver
from number_states_solvers import ExcitedStatesSolver
from excited_amplitudes_solver import ExcitedAmplitudesSolver
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5

def main_gs_es():
    gs_solver = GroundStateSolver(h)
    gs_solver.run()

    es_solver = ExcitedStatesSolver(h)
    es_solver.run()

def main_amp(method='exact'):
    amp_solver = ExcitedAmplitudesSolver(h)
    amp_solver.run()

if __name__ == '__main__': 
    h = get_lih_hamiltonian(3.0)
    # suffix = '_sv'

    initialize_hdf5()
    main_gs_es()
    main_amp()
