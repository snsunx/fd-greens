"""
=========================================
Parameters (:mod:`fd_greens.main.params`)
=========================================
"""
import numpy as np

from qiskit.circuit import Barrier
from qiskit.extensions import SwapGate, CCXGate, XGate, HGate, UnitaryGate

from .qubit_indices import QubitIndices

# Constants
HARTREE_TO_EV = 27.211386245988
c = 1

# Variables for compilation
basis_gates = ["u3", "cz", "swap", "cp"]
# swap_direcs_round1 = {(0, 1): [['left', 'left'], ['right', 'right', 'left'], ['right', 'left'], ['right', 'right'], ['right']],
#                      (1, 0): [['left'], ['right', 'right'], ['left', 'left', 'left'], ['right', 'right', 'left'], ['right']]}
swap_direcs_round1 = [
    ["left", "left"],
    ["right", "right", "left"],
    ["right", "left"],
    ["right", "right"],
    ["right"],
]
# swap_direcs_round2 = {(0, 1): [['left', None], [None, None, None], [None, None], [], [None]],
#                      (1, 0): [[None], [], [None, None, None], [None, None, None], [None]]}
swap_direcs_round2 = [["left", None], [None, None, None], [None, None], [], [None]]
swap_direcs_tomo = {
    (0, 1): [[None, None], [None, None, None], [None, None], [], ["right"]]
}

# Qubit indices for Green's functions
eu_inds = QubitIndices(["1101", "0111"])
ed_inds = QubitIndices(["1110", "1011"])
hu_inds = QubitIndices(["0100", "0001"])
hd_inds = QubitIndices(["1000", "0010"])
e_inds = {"u": eu_inds, "d": ed_inds}
h_inds = {"u": hu_inds, "d": hd_inds}

# Qubit indices for charge-charge correlation functions
singlet_inds = QubitIndices(["0011", "0110", "1001", "1100"])
triplet_inds = QubitIndices(["0101", "1010"])

# CCX gate decomposition
ccx_inst_tups = [
    (SwapGate(), [1, 2], []),
    (XGate(), [0], []),
    (XGate(), [2], []),
    # (Barrier(4), [0, 1, 2, 3], []),
    (CCXGate(ctrl_state="00"), [0, 2, 1], []),
    # (Barrier(4), [0, 1, 2, 3], []),
    (XGate(), [0], []),
    (XGate(), [2], []),
    (SwapGate(), [1, 2], []),
]

cxc_inst_tups = [
    (XGate(), [0], []),
    (XGate(), [2], []),
    (Barrier(4), [0, 1, 2, 3], []),
    (CCXGate(ctrl_state="00"), [0, 2, 1], []),
    (Barrier(4), [0, 1, 2, 3], []),
    (XGate(), [0], []),
    (XGate(), [2], []),
]

CCZGate = UnitaryGate(np.array([[1, 0], [0, -1]])).control(2, ctrl_state="11")
# CCiXGate = UnitaryGate(1j*np.array([[0, 1], [1, 0]])).control(2, ctrl_state='11')
# C0C0iXGate = UnitaryGate(1j*np.array([[0, 1], [1, 0]])).control(2, ctrl_state='00')
C0C0iXGate = UnitaryGate(
    [
        [0, 0, 0, 0, 1j, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1j, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)
