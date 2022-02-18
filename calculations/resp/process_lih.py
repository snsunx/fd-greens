import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from greens_function import GreensFunction
from utils import get_lih_hamiltonian
from params import HARTREE_TO_EV
from amplitudes_solvers_new import EHAmplitudesSolver

def main_amp(method='exact'):
    amp_solver = EHAmplitudesSolver(h, h5fname=h5fname, method=method, suffix=suffix)
    amp_solver.process_all()

def main_greens(spec_func=True, self_e=True):
    greens_func = GreensFunction(h5fname, suffix=suffix)
    if spec_func: greens_func.spectral_function(omegas, eta)
    if self_e: greens_func.self_energy(omegas, eta)

if __name__ == '__main__':
    h = get_lih_hamiltonian(3.0)
    omegas = np.arange(-20, 10, 0.1)
    eta = 0.02 * HARTREE_TO_EV

    main_amp('exact')
    main_greens()
