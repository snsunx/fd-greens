import sys
sys.path.append('../../src/')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from response_function import ResponseFunction
from params import HARTREE_TO_EV

resp_func = ResponseFunction('lih_exc')
omegas = np.arange(-30, 30, 0.1)
eta = 0.01 * HARTREE_TO_EV

resp_func.response_function(omegas, 0, 0, eta)
resp_func.response_function(omegas, 1, 1, eta)
resp_func.cross_section(omegas, eta)
