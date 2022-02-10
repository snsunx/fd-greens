import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from greens_function import GreensFunction
from params import HARTREE_TO_EV

from run_eh import amp_solver

amp_solver.initialize()
amp_solver.process_all()

greens_func = GreensFunction('lih_expt_qasm')
omegas = np.arange(-30, 30, 0.1)
eta = 0.02 * HARTREE_TO_EV

greens_func.spectral_function(omegas, eta)
greens_func.self_energy(omegas, eta)
