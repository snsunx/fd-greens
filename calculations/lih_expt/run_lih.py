import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')

from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from amplitudes_solvers_new import EHAmplitudesSolver
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5

def main_gs_eh():
    gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance, 
                              method='exact', h5fname=h5fname)
    eh_solver = EHStatesSolver(h, ansatz_func_e=build_ansatz_e, ansatz_func_h=build_ansatz_h, 
                           q_instance=q_instance, method='exact', h5fname=h5fname)
    gs_solver.run()
    eh_solver.run()

def main_amp(build=True, run=True, method='exact'):
    amp_solver = EHAmplitudesSolver(h, q_instance=q_instance, method=method, h5fname=h5fname, suffix=suffix)
    
    if build: amp_solver.build_all()
    if run: amp_solver.run_all()

if __name__ == '__main__': 
    h = get_lih_hamiltonian(3.0)
    q_instance = get_quantum_instance('qasm')
    h5fname = 'lih_3A'
    suffix = '_1'

    initialize_hdf5(h5fname)
    main_gs_eh()
    main_amp(build=True, method='tomo')
