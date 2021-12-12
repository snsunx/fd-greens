from qubit_indices import QubitIndices
from typing import Union 

import numpy as np

from qiskit.circuit import Barrier
from qiskit.extensions import UnitaryGate, SwapGate


HARTREE_TO_EV = 27.211386245988
c = 1

basis_gates = ['u3', 'cz', 'swap', 'cp']
swap_direcs_round1 = {(0, 1): [['left', 'left'], ['right', 'right', 'left'], ['right', 'left'], ['right', 'right'], ['right']],
                      (1, 0): [['left'], ['right', 'right'], ['left', 'left', 'left'], ['right', 'right', 'left'], ['right']]}
swap_direcs_round2 = {(0, 1): [['left', None], [None, None, None], [None, None], [], [None]],
                      (1, 0): [[None], [], [None, None, None], [None, None, None], [None]]}

eu_inds = QubitIndices(['1101', '0111'])
ed_inds = QubitIndices(['1110', '1011'])
hu_inds = QubitIndices(['0100', '0001'])
hd_inds = QubitIndices(['1000', '0010'])

singlet_inds = QubitIndices(['0011', '0110', '1001', '1100'])
triplet_inds = QubitIndices(['0101', '1010'])

iX = np.array([[0, 1j], [1j, 0]])
ccx_data = [(SwapGate(), [1, 2]), 
            (Barrier(4), [0, 1, 2, 3]), 
            (UnitaryGate(iX).control(2), [0, 2, 1]), 
            (Barrier(4), [0, 1, 2, 3]),
            (SwapGate(), [1, 2])]