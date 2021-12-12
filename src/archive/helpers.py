from typing import Union 

import os
import numpy as np
import h5py

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import Barrier
from qiskit.extensions import UnitaryGate, SwapGate

from hamiltonians import MolecularHamiltonian
#from ground_state_solvers import GroundStateSolver
#from number_states_solvers import EHStatesSolver, ExcitedStatesSolver
#from amplitudes_solvers import EHAmplitudesSolver, ExcitedAmplitudesSolver
