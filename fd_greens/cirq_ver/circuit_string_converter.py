"""
====================================================================
Circuit String Converter (:mod:`fd_greens.circuit_string_converter`)
====================================================================
"""

import re
from typing import List, Sequence

import numpy as np
import cirq

from .transpilation import C0C0iXGate
from .utilities import get_gate_counts
from .parameters import WRAP_Z_AROUND_ITOFFOLI


class CircuitStringConverter:
    """Converter between Cirq circuits and Qtrl strings."""

    def __init__(self, qubits: Sequence[cirq.Qid], offset: int = 4) -> None:
        """Initializes a ``CircuitStringConverter`` object.
        
        Args:
            qubits: The qubits in the circuit.
            offset: The index offset on the Berkeley device. Defaults to 4.
        """
        self.qubits = qubits
        self.offset = offset

    def _qstring_to_qubits(self, qstring: str) -> List[cirq.Qid]:
        """Converts Qtrl qubit string to Cirq qubits."""
        inds = re.findall("(\d)", qstring)
        qubits = [self.qubits[int(x) - self.offset] for x in inds]
        return qubits

    def _qubits_to_qstring(self, qubits: Sequence[cirq.Qid]) -> str:
        """Converts Cirq qubits to Qtrl qubit string."""
        if len(qubits) == 1:
            qstring = "Q" + str(qubits[0].x + self.offset)
        elif len(qubits) == 2:
            indices = sorted([q.x + self.offset for q in qubits], reverse=True)
            qstring = f"C{indices[0]}T{indices[1]}"
        elif len(qubits) == 3:
            indices = [q.x + self.offset for q in qubits]
            qstring = f"C{indices[0]}C{indices[1]}T{indices[2]}"
        else:
            raise ValueError(f"A gate acting on {len(qubits)} qubits cannot be converted to Qtrl string.")
        return qstring

    def _gstring_to_gate(self, gstring: str) -> cirq.Gate:
        """Converts Qtrl gate string to Cirq gates."""
        if re.findall("\d+", gstring) == []:  # No parameter, multi-qubit gate
            if gstring == "CZ":
                gate = cirq.CZ
            elif gstring == "CS":
                gate = cirq.CZPowGate(exponent=0.5)
            elif gstring == "CSD":
                gate = cirq.CZPowGate(exponent=-0.5)
            elif gstring == "TOF":
                gate = C0C0iXGate()
            elif gstring == "SWAP":
                gate = cirq.SWAP
            else:
                raise NotImplementedError(f"Parameter-free gate can only be CZ, CS, CSD or TOF. Got {gstring}.")
        else:  # Has parameter, single-qubit gate
            gate_name = gstring[0]
            exponent = float(gstring[1:]) / 180
            if gate_name == "X":
                gate = cirq.XPowGate(exponent=exponent)
            elif gate_name == "Z":
                gate = cirq.ZPowGate(exponent=exponent)
            else:
                raise NotImplementedError(f"Parametrized gate can only be X or Z. Got {gate_name}.")
        return gate

    def _gate_to_gstring(self, gate: cirq.Gate) -> str:
        """Converts Cirq gates to Qtrl gate string."""
        # print(f'{gate = }')
        # if isinstance(gate, (cirq.XPowGate, cirq.ZPowGate)):
        #     gname = gate.__class__.__name__[0]
        #     angle = str(gate.exponent * 180.0)
        #     gstring = gname + angle
        if isinstance(gate, cirq.XPowGate):
            assert abs(gate.exponent - 0.5) < 1e-8
            gstring = 'X90'
        elif isinstance(gate, cirq.ZPowGate):
            angle = gate.exponent * 180.0
            gstring = f'Z{angle}'
        elif isinstance(gate, cirq.CZPowGate):
            if abs(abs(gate._exponent) - 1.0) < 1e-8:
                gstring = 'CZ'
            elif abs(gate._exponent - 0.5) < 1e-8:
                gstring = 'CS'
            elif abs(gate._exponent + 0.5) < 1e-8:
                gstring = 'CSD'
            else:
                raise ValueError(f"CZPowGate must have exponent 1.0, 0.5 or -0.5. Got {gate._exponent}.")
        elif isinstance(gate, C0C0iXGate):
            assert isinstance(gate, C0C0iXGate) == (type(gate) is C0C0iXGate)
            gstring = "TOF"
        elif str(gate) == "SWAP":
            # TODO: This could be raised as a warning.
            print("Warning: SWAP gate converted to Qtrl string.")
            gstring = "SWAP"
        else:
            raise NotImplementedError(f"Conversion of {str(gate)} to Qtrl string is not supported.")
        return gstring

    def convert_strings_to_circuit(self, qtrl_strings: List[List[str]]) -> cirq.Circuit:
        """Converts Qtrl strings to a Cirq circuit.

        The Qtrl strings are saved in HDF5 files. To convert the Qtrl strings saved in ``lih_3A.h5`` to
        a Cirq circuit, we need to ::

            import sys
            sys.path.append('path/to/fd_greens')

            import h5py
            import json
            import cirq

            from fd_greens import CircuitStringConverter

            qubits = cirq.LineQubit.range(4)
            converter = CircuitStringConverter(qubits)

            h5file = h5py.File('lih_3A.h5', 'r')
            qtrl_strings = json.loads(h5file['circ0d/transpiled'][()])
            circuit = converter.convert_strings_to_circuit(qtrl_strings)

        Args:
            qtrl_strings: Qtrl strings.
            
        Returns:
            circuit: A Cirq circuit corresponding to the Qtrl strings. 
        """
        # Flatten the qtrl strings if they are not flat.
        # if isinstance(qtrl_strings[0], list):
        #     qtrl_strings = [y for x in qtrl_strings for y in x]
        circuit = cirq.Circuit()

        # Iterate over each qtrl string. The qstr, gstr order depends on whether
        # the gate is 1q or multi-q gate.
        for strings_moment in qtrl_strings:
            ops_moment = []
            for string in strings_moment:
                if string[0] == "Q":  # single-qubit gate
                    qstring, gstring = string.split("/")
                else:  # multi-qubit gate
                    gstring, qstring = string.split("/")
                # print(f'{qstr = }')
                # print(f'{gstr = }')
                qubits = self._qstring_to_qubits(qstring)
                gate = self._gstring_to_gate(gstring)
                ops_moment.append(gate(*qubits))
            circuit.append(cirq.Moment(ops_moment))

        return circuit

    def convert_circuit_to_strings(self, circuit: cirq.Circuit) -> List[List[str]]:
        """Converts a Cirq circuit to Qtrl strings.
        
        Args:
            circuit: A Cirq circuit.
            
        Returns:
            qtrl_strings: The Qtrl strings corresponding to the Cirq circuit.
        """

        qtrl_strings = []

        for moment in circuit:

            # TODO: Write this better with check statements.
            gate_counts = [get_gate_counts(cirq.Circuit(moment), num_qubits=i) for i in [1, 2, 3]]

            if np.count_nonzero(gate_counts) != 1:
                print(gate_counts)
                print(moment)
                exit()

            strings_moment = []
            for op in moment:
                gate = op.gate
                qubits = op.qubits

                # Ignore identity and measurement gates.
                if isinstance(op.gate, (cirq.IdentityGate, cirq.MeasurementGate)):
                    continue

                gstring = self._gate_to_gstring(gate)
                qstring = self._qubits_to_qstring(qubits)

                if len(qubits) == 1:
                    qtrl_string = qstring + "/" + gstring
                else:
                    qtrl_string = gstring + "/" + qstring
                strings_moment.append(qtrl_string)
            
            if strings_moment != []:
                if WRAP_Z_AROUND_ITOFFOLI and len(strings_moment) == 1 and strings_moment[0][:3] == 'TOF':
                    qtrl_strings.append(["Q5/Z-20.67"])
                qtrl_strings.append(strings_moment)
                if WRAP_Z_AROUND_ITOFFOLI and len(strings_moment) == 1 and strings_moment[0][:3] == 'TOF':
                    qtrl_strings.append(["Q5/Z20.67"])
            
        return qtrl_strings
