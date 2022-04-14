import re
from typing import List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.extensions import RXGate, RYGate, RZGate, CZGate, CPhaseGate, CCXGate

from .circuit_utils import create_circuit_from_inst_tups

def qargs_from_str(qstr: str, offset: int = 4) -> List[int]:
    """Converts a qtrl qargs string to qargs."""
    qargs = re.findall('(\d)', qstr)
    qargs = [int(x) - offset for x in qargs]
    return qargs

def gate_from_str(gstr: str) -> Gate:
    """Converts a qtrl gate string to gate."""
    if re.findall('\d+', gstr) == []:
        if gstr == 'CZ': 
            gate = CZGate()
        elif gstr == 'CS':
            gate = CPhaseGate(np.pi/2)
        elif gstr == 'CSD':
            gate = CPhaseGate(-np.pi/2)
        elif gstr == 'TOF':
            # TODO: CCXGate is just a placeholder.
            gate = CCXGate()
    else:
        gname = gstr[0]
        angle = float(gstr[1:])
        if gname == 'X':
            gate = RXGate(angle)
        elif gname == 'Y':
            gate = RYGate(angle)
        elif gname == 'Z':
            gate = RZGate(angle)
    return gate

def create_circuit_from_qtrl_strings(qtrl_strings: List[List[str]]) -> QuantumCircuit:
    """Creates a circuit from qtrl strings.
    
    Args:
        qtrl_strings: The qtrl strings of the circuit.
        
    Returns:
        circ: A Qiskit circuit built from the qtrl strings.
    """
    qtrl_strings = [y for x in qtrl_strings for y in x]
    inst_tups = []
    
    for s in qtrl_strings:
        if s[0] == 'Q':
            qstr, gstr = s.split('/')
        else:
            gstr, qstr = s.split('/')
        qargs = qargs_from_str(qstr)
        gate = gate_from_str(gstr)
        inst_tups += [(gate, qargs, [])]

    circ = create_circuit_from_inst_tups(inst_tups)
    return circ

