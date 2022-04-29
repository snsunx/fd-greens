"""
============================================================
Interface Utilities (:mod:`fd_greens.utils.interface_utils`)
============================================================
"""

import re
from typing import List, Sequence
import json

import numpy as np
import cirq

from .transpilation import C0C0iXGate


class CircuitStringConverter:
    """Converter between circuits and their string form."""

    def __init__(self, qubits: Sequence[cirq.Qid], offset: int = 4) -> None:
        """Initializes a ``CircuitStringConverter`` object.
        
        Args:
            qubits: The qubits in the circuit.
            offset: The index offset on the Berkeley device.
        """
        self.qubits = qubits
        self.offset = offset

    def _qstr_to_qubits(self, qstr: str) -> List[cirq.Qid]:
        """Qtrl qubit string to Cirq qubits."""
        inds = re.findall("(\d)", qstr)
        qubits = [self.qubits[int(x) - self.offset] for x in inds]
        return qubits

    def _qubits_to_qstr(self, qubits: Sequence[cirq.Qid]) -> str:
        """Cirq qubits to qtrl qubit string."""
        if len(qubits) == 1:
            qstr = "Q" + str(qubits[0].x + self.offset)
        elif len(qubits) == 2:
            inds = sorted([q.x + self.offset for q in qubits], reverse=True)
            qstr = f"C{inds[0]}T{inds[1]}"
        elif len(qubits) == 3:
            inds = [q.x + self.offset for q in qubits]
            qstr = f"C{inds[0]}C{inds[1]}T{inds[2]}"
        else:
            raise ValueError("A gate can only act on up to 3 qubits.")

        return qstr

    def _gstr_to_gate(self, gstr: str) -> cirq.Gate:
        """Qtrl gate string to Cirq gate.
        
        Args:
            gstr: A qtrl gate string.
            
        Returns:
            gate: The Cirq gate corresponding to the qtrl gate string. 
        """
        if re.findall("\d+", gstr) == []: # No parameter, multi-q gate
            if gstr == "CZ":
                gate = cirq.CZ
            elif gstr == "CS":
                gate = cirq.CZPowGate(exponent=0.5)
            elif gstr == "CSD":
                gate = cirq.CZPowGate(exponent=-0.5)
            elif gstr == "TOF":
                gate = C0C0iXGate
        else: # Has parameter, 1q gate
            gname = gstr[0]
            exponent = float(gstr[1:]) / 180
            if gname == "X":
                gate = cirq.XPowGate(exponent=exponent)
            elif gname == "Z":
                gate = cirq.ZPowGate(exponent=exponent)
        return gate

    def _gate_to_gstr(self, gate: cirq.Gate) -> str:
        """Cirq gate to qtrl gate string.
        
        Args:
            gate: A Cirq gate.
            
        Returns:
            gstr: The qtrl gate string corresponding to the Cirq gate.
        """
        if isinstance(gate, cirq.SingleQubitGate):  # Rx or Rz
            gname = gate.__class__.__name__[0]
            angle = str(gate.exponent * 180.0)
            gstr = gname + angle
        elif isinstance(gate, cirq.CZPowGate):
            if gate._exponent == 1.0:
                gstr = "CZ"
            elif gate._exponent == 0.5:
                gstr = "CS"
            elif gate._exponent == -0.5:
                gstr = "CSD"
            else:
                raise ValueError("CZPowGate must have exponent 1.0, 0.5 or -0.5.")
        elif isinstance(gate, C0C0iXGate):
            gstr = "TOF"
        else:
            raise TypeError("")
        return gstr

    def convert_strings_to_circuit(self, strings: List[List[str]]) -> cirq.Circuit:
        """Converts qtrl strings to a Qiskit circuit.
        
        Args:
            strings: Qtrl strings.
            
        Returns:
            circuit: A Cirq circuit corresponding to the qtrl strings.
        """
        # Flatten the qtrl strings if they are not flat.
        if isinstance(strings[0], list):
            strings = [y for x in strings for y in x]
        circuit = cirq.Circuit()

        # Iterate over each qtrl string. The qstr, gstr order depends on whether
        # the gate is 1q or multi-q gate.
        for s in strings:
            if s[0] == "Q":  # 1q gate
                qstr, gstr = s.split("/")
            else:  # multi-q gate
                gstr, qstr = s.split("/")
            # print(f'{qstr = }')
            # print(f'{gstr = }')
            qubits = self._qstr_to_qubits(qstr)
            gate = self._gstr_to_gate(gstr)
            circuit.append(gate(*qubits))

        return circuit

    def convert_circuit_to_strings(self, circuit: cirq.Circuit) -> List[List[str]]:
        """Converts a Cirq circuit to qtrl strings.
        
        Args:
            circuit: A Cirq circuit.
            
        Returns:
            strings: The qtrl strings corresponding to the Cirq circuit.
        """
        # circ = remove_instructions_in_circuit(circ, ["measure", "barrier"])
        strings = []

        for moment in circuit:
            strings_moment = []
            for op in moment:
                gate = op.gate
                qubits = op.qubits

                gstr = self._gate_to_gstr(gate)
                qstr = self._qubits_to_qstr(qubits)

                if len(qubits) == 1:
                    qtrl_str = qstr + "/" + gstr
                else:
                    qtrl_str = gstr + "/" + qstr
                strings_moment.append(qtrl_str)
            strings.append(strings_moment)
        return strings
