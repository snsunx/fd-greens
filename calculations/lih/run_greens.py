import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from greens_function import GreensFunction
from params import HARTREE_TO_EV

greens_func = GreensFunction('lih', 'eh_tomo')
omegas = np.arange(-30, 30, 0.1)
eta = 0.02 * HARTREE_TO_EV

greens_func.spectral_function(omegas, eta)
greens_func.self_energy(omegas, eta)
