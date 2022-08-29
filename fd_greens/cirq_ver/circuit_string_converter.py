"""
====================================================================
Circuit String Converter (:mod:`fd_greens.circuit_string_converter`)
====================================================================
"""

import re
from typing import List, Sequence, Union
import warnings

import numpy as np
import cirq
import h5py
import json

from .transpilation import C0C0iXGate
from .general_utils import get_gate_counts, unitary_equal
from .parameters import CHECK_CIRCUITS, CircuitConstructionParameters, QUBIT_OFFSET


class CircuitStringConverter:
    """Converter between Cirq circuits and Qtrl strings."""

    def __init__(self, qubits: Sequence[cirq.Qid]) -> None:
        """Initializes a ``CircuitStringConverter`` object.
        
        Args:
            qubits: The qubits in the circuit.
        """
        self.qubits = qubits

        self.parameters = CircuitConstructionParameters()

    def _qstring_to_qubits(self, qstring: str) -> List[cirq.Qid]:
        """Converts Qtrl qubit string to Cirq qubits."""
        inds = re.findall("(\d)", qstring)
        qubits = [self.qubits[int(x) - QUBIT_OFFSET] for x in inds]
        return qubits

    def _qubits_to_qstring(self, qubits: Sequence[cirq.Qid]) -> str:
        """Converts Cirq qubits to Qtrl qubit string."""
        if len(qubits) == 1:
            qstring = "Q" + str(qubits[0].x + QUBIT_OFFSET)
        elif len(qubits) == 2:
            indices = sorted([q.x + QUBIT_OFFSET for q in qubits], reverse=True)
            qstring = f"C{indices[0]}T{indices[1]}"
        elif len(qubits) == 3:
            indices = [q.x + QUBIT_OFFSET for q in qubits]
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
            elif gstring == "CP": # CPhase gate after iToffoli
                gate = cirq.I
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
                if abs(abs(exponent) - self.parameters.ITOFFOLI_Z_ANGLE / 180.0) < 1e-8:
                    gate = cirq.I # Z gates around iToffoli
                else:
                    gate = cirq.ZPowGate(exponent=exponent)
            else:
                raise NotImplementedError(f"Parametrized gate can only be X or Z. Got {gate_name}.")
        return gate

    def _gate_to_gstring(self, gate: cirq.Gate) -> str:
        """Converts Cirq gates to Qtrl gate string."""
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
            gstring = "TOF"
        elif str(gate) == "SWAP":
            warnings.warn("SWAP gate converted to Qtrl string.")
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
        circuit = cirq.Circuit()

        # Iterate over qtrl_strings and construct the circuit from each moment.
        for strings_moment in qtrl_strings:
            ops_moment = []
            for string in strings_moment:
                if string[0] == 'Q':  # single-qubit gate
                    qstring, gstring = string.split("/")
                else:  # multi-qubit gate
                    gstring, qstring = string.split("/")
                
                qubits = self._qstring_to_qubits(qstring)
                gate = self._gstring_to_gate(gstring)
                if gate != cirq.I: # Not Z and CP around iToffoli, which were set as cirq.I
                    ops_moment.append(gate(*qubits))
            
            if ops_moment != []:
                circuit.append(cirq.Moment(ops_moment))

        return circuit

    def convert_circuit_to_strings(self, circuit: cirq.Circuit) -> List[List[str]]:
        """Converts a Cirq circuit to Qtrl strings.

        This function optionally wraps around iToffoli gates with Z and CPhase gates.
        
        Args:
            circuit: A Cirq circuit.
            
        Returns:
            qtrl_strings: The Qtrl strings corresponding to the Cirq circuit.
        """
        qtrl_strings = []

        i_moment = 0
        adjustments = []
        for moment in circuit:
            # Check the moment contains only one type of gates.
            if CHECK_CIRCUITS:
                gate_counts = [get_gate_counts(cirq.Circuit(moment), num_qubits=i) for i in [1, 2, 3]]
                if np.count_nonzero(gate_counts) != 1:
                    raise ValueError("A moment can only consist of one type of gates.")

            strings_moment = []
            for op in moment:
                # Ignore identity and measurement gates.
                if isinstance(op.gate, (cirq.IdentityGate, cirq.MeasurementGate)):
                    continue

                # Combine gate string and qubit string to a single qtrl string.
                gstring = self._gate_to_gstring(op.gate)
                qstring = self._qubits_to_qstring(op.qubits)
                if len(op.qubits) == 1:
                    qtrl_string = qstring + "/" + gstring
                else:
                    qtrl_string = gstring + "/" + qstring
                
                # Wrap Z and CPhase around iToffoli.
                if self.parameters.WRAP_Z_AROUND_ITOFFOLI:
                    angle = self.parameters.ITOFFOLI_Z_ANGLE
                    if qtrl_string == 'TOF/C4C6T5':
                        adjustments.append((i_moment, [f'Q5/Z-{angle}']))
                        adjustments.append((i_moment + 1, [f'Q5/Z{angle}']))
                        adjustments.append((i_moment + 1, ['CP/C7T6']))
                
                strings_moment.append(qtrl_string)

            if strings_moment != []:
                i_moment += 1
                qtrl_strings.append(strings_moment)

        # Insert adjustment gates to qtrl_strings.
        moment_offset = 0
        for i, strings_moment in adjustments:
            qtrl_strings.insert(i + moment_offset, strings_moment)
            moment_offset += 1
        
        return qtrl_strings

    def adapt_to_hardware(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Adapts a circuit to hardware by converting to native gates.
        
        This function handles adjusting CS/CSD gates on specific qubit pairs, and splitting 
        simultaneous CZ gates onto different moments.

        Args:
            circuit: The circuit on which gates are to be adapted.
        
        Returns:
            circuit_new: The new circuit after gate adaptation.
        """
        # print("Before adapting to hardware")
        # print_circuit(circuit)
        circuit_new = cirq.Circuit()

        i_moment = 0
        adjustments = []
        for moment in circuit:
            moment_new = []
            for operation in moment:
                gate = operation.gate
                qubits = operation.qubits

                if self.parameters.ADJUST_CS_CSD:
                    if isinstance(gate, cirq.CZPowGate):
                        is_cs_on_45 = abs(gate.exponent - 0.5) < 1e-8 and set(qubits) == set(self.qubits[:2])
                        is_csd_on_56 = abs(gate.exponent + 0.5) < 1e-8 and set(qubits) == set(self.qubits[1:3])
                        is_cs_on_67 = abs(gate.exponent - 0.5) < 1e-8 and set(qubits) == set(self.qubits[2:4])
                        if is_cs_on_45 or is_csd_on_56 or is_cs_on_67:
                            moment_new.append(cirq.CZ(*qubits))
                            adjustments.append((i_moment, cirq.CZPowGate(exponent=-gate.exponent)(*qubits)))
                        else:
                            moment_new.append(operation)
                    else:
                        moment_new.append(operation)
                else:
                    moment_new.append(operation)

            # print(f"{moment_new = }")
            if self.parameters.SPLIT_SIMULTANEOUS_CZS:
                is_cz_pow_gates = [isinstance(op.gate, cirq.CZPowGate) for op in moment_new]
                if is_cz_pow_gates == [True, True]:
                    i_moment += 2
                    circuit_new.append(cirq.Moment(moment_new[0]))
                    circuit_new.append(cirq.Moment(moment_new[1]))
                else:
                    i_moment += 1
                    circuit_new.append(cirq.Moment(moment_new))
            else:
                i_moment += 1
                circuit_new.append(cirq.Moment(moment_new))

        # Insert adjustment gates.
        moment_offset = 0
        for i, gate in adjustments:
            circuit_new.insert(i + moment_offset, gate)
            moment_offset += 1

        # print("After adapting to hardware")
        # print_circuit(circuit_new)
        if CHECK_CIRCUITS:
            assert unitary_equal(circuit, circuit_new)

        return circuit_new

    def load_circuit(self, h5fname: Union[str, h5py.File], dsetname: str) -> cirq.Circuit:
        """Loads a circuit from an HDF5 file."""
        if isinstance(h5fname, str):
            h5file = h5py.File(h5fname + '.h5', 'r')
        else:
            h5file = h5fname
        
        qtrl_strings = json.loads(h5file[dsetname][()])
        circuit = self.convert_strings_to_circuit(qtrl_strings)

        if isinstance(h5fname, str):
            h5file.close()
        return circuit
    
    def save_circuit(
        self,
        h5fname: Union[str, h5py.File],
        dsetname: str, 
        circuit: cirq.Circuit,
        return_dataset: bool = True
    ) -> None:
        """Saves a circuit to an HDF5 file."""
        if return_dataset:
            assert isinstance(h5fname, h5py.File)
        if isinstance(h5fname, str):
            h5file = h5py.File(h5fname + '.h5', 'r+')
        else:
            h5file = h5fname
        
        qtrl_strings = self.convert_circuit_to_strings(circuit)
        if dsetname in h5file:
            # print("dsetname in h5file")
            del h5file[dsetname]
            h5file[dsetname] = json.dumps(qtrl_strings)
        else:
            # print("dsetname not in h5file")
            h5file.create_dataset(dsetname, data=json.dumps(qtrl_strings))
        
        if isinstance(h5fname, str):
            h5file.close()

        if return_dataset:
            return h5file[dsetname]
