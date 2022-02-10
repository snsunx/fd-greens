import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')

from ground_state_solvers import GroundStateSolver
from number_states_solvers import EHStatesSolver
from amplitudes_solvers_new import EHAmplitudesSolver
from ansatze import build_ansatz_gs, build_ansatz_e, build_ansatz_h
from utils import get_lih_hamiltonian, get_quantum_instance, initialize_hdf5

h = get_lih_hamiltonian(1.6)
q_instance = get_quantum_instance('qasm')
h5fname = 'lih_expt_' + q_instance.backend.name()[:4]

gs_solver = GroundStateSolver(h, ansatz_func=build_ansatz_gs, q_instance=q_instance, 
                              method='exact', h5fname=h5fname)
es_solver = EHStatesSolver(h, ansatz_func_e=build_ansatz_e, ansatz_func_h=build_ansatz_h, 
                           q_instance=q_instance, method='exact', h5fname=h5fname)
if q_instance.backend.name()[:4] == 'stat': 
    amp_solver = EHAmplitudesSolver(h, q_instance=q_instance, method='exact', h5fname=h5fname)
elif q_instance.backend.name()[:4] == 'qasm':
    amp_solver = EHAmplitudesSolver(h, q_instance=q_instance, method='tomo', h5fname=h5fname)

if __name__ == '__main__': 
    initialize_hdf5(h5fname)
    gs_solver.run()
    es_solver.run()
    amp_solver.initialize()
    amp_solver.build_all()
    amp_solver.run_all()
