"""
============================================================
Interface Utilities (:mod:`fd_greens.utils.interface_utils`)
============================================================
"""

import re
from typing import List
import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Qubit
from qiskit.extensions import RXGate, RYGate, RZGate, CZGate, CPhaseGate

from ..main.params import C0C0iXGate
from .circuit_utils import create_circuit_from_inst_tups, remove_instructions_in_circuit
from .general_utils import circuit_equal


def qstr_to_qargs(qstr: str, offset: int = 4) -> List[int]:
    """Qtrl qargs string to Qiskit qargs.
    
    Args:
        qstr: The qtrl qargs string, e.g. "Q6", "C5T6".
        offset: The offset of the qubits on the Berkeley device. Default to 4.
    
    Returns:
        qargs: The Qiskit qargs object.
    """
    qargs = re.findall("(\d)", qstr)
    qargs = [int(x) - offset for x in qargs]
    return qargs


def qargs_to_qstr(qargs: List[Qubit], gate: Gate, offset: int = 4) -> str:
    """Qiskit qargs to qtrl qargs string.
    
    Args:
        qargs: A Qiskit qargs object.
        gate: A quantum gate.
        offset: Index of the first qubit in use on the Berkeley device.
    
    Returns:
        qstr: The qtrl qargs string.
    """
    assert gate.name in ["rx", "rz", "cz", "cp", "c0c0ix"]
    if gate.name in ["rx", "rz"]:
        assert len(qargs) == 1
        qstr = "Q" + str(qargs[0]._index + offset)
    elif gate.name in ["cz", "cp"]:
        assert len(qargs) == 2
        qnum = list(reversed([q._index + offset for q in qargs]))
        qstr = f"C{qnum[0]}T{qnum[1]}"
    elif gate.name == "c0c0ix":
        assert len(qargs) == 3
        qnum = [q._index + offset for q in qargs]
        qstr = f"C{qnum[0]}C{qnum[1]}T{qnum[2]}"
    # print(f"{qstr=}")
    return qstr


def gstr_to_gate(gstr: str) -> Gate:
    """qtrl gate string to Qiskit gate.
    
    Args:
        gstr: A qtrl gate string.
        
    Returns:
        gate: The Qiskit gate corresponding to the qtrl gate string. 
    """
    if re.findall("\d+", gstr) == []:
        if gstr == "CZ":
            gate = CZGate()
        elif gstr == "CS":
            gate = CPhaseGate(np.pi / 2)
        elif gstr == "CSD":
            gate = CPhaseGate(-np.pi / 2)
        elif gstr == "TOF":
            gate = C0C0iXGate
    else:
        gname = gstr[0]
        angle = float(gstr[1:]) / 180 * np.pi
        if gname == "X":
            gate = RXGate(angle)
        elif gname == "Y":
            gate = RYGate(angle)
        elif gname == "Z":
            gate = RZGate(angle)
    return gate


def gate_to_gstr(gate: Gate) -> str:
    """Qiskit gate to qtrl gate string.
    
    Args:
        gate: A Qiskit gate.
        
    Returns:
        gstr: The qtrl gate string corresponding to the Qiskit gate.
    """
    assert gate.name in ["rx", "rz", "cz", "cp", "c0c0ix"]
    if gate.name in ["rx", "rz"]:
        gname = gate.name[-1].upper()
        angle = gate.params[0] / np.pi * 180
        gstr = gname + str(angle)
    elif gate.name == "cz":
        gstr = "CZ"
    elif gate.name == "cp":
        if abs(gate.params[0] - np.pi / 2) < 1e-8:
            gstr = "CS"
        elif abs(gate.params[0] + np.pi / 2) < 1e-8:
            gstr = "CSD"
    elif gate.name == "c0c0ix":
        gstr = "TOF"
    return gstr


def qtrl_strings_to_qiskit_circuit(qtrl_strs: List[List[str]]) -> QuantumCircuit:
    """Converts qtrl strings to a Qiskit circuit.
    
    Args:
        qtrl_strings: A list of qtrl strings.
        
    Returns:
        circ: A Qiskit circuit corresponding to the qtrl strings.
    """
    # TODO: Now implemented as a list. Should be a list of lists.
    if isinstance(qtrl_strs[0], list):
        qtrl_strs = [y for x in qtrl_strs for y in x]
    inst_tups = []

    for s in qtrl_strs:
        if s[0] == "Q":
            qstr, gstr = s.split("/")
        else:
            gstr, qstr = s.split("/")
        qargs = qstr_to_qargs(qstr)
        gate = gstr_to_gate(gstr)
        inst_tups += [(gate, qargs, [])]

    circ = create_circuit_from_inst_tups(inst_tups)
    return circ


def qiskit_circuit_to_qtrl_strings(circ: QuantumCircuit) -> List[str]:
    """Converts a Qiskit circuit to qtrl strings.

    The only gates that can be converted are ``"rx"``, ``"rz"``, ``"cz"``,
    ``"cp"``, ``"c0c0ix"``.
    
    Args:
        circ: A Qiskit circuit.
        
    Returns:
        qtrl_strs: The qtrl strings corresponding to the Qiskit circuit.
    """
    circ = remove_instructions_in_circuit(circ, ["measure", "barrier"])
    qtrl_strs = []

    for gate, qargs, _ in circ.data:
        # print(f"{gate.name=}")
        gstr = gate_to_gstr(gate)
        qstr = qargs_to_qstr(qargs, gate)
        # print(gstr, qstr)
        if gate.name in ["rx", "rz"]:
            qtrl_str = qstr + "/" + gstr
        elif gate.name in ["cz", "cp", "c0c0ix"]:
            qtrl_str = gstr + "/" + qstr
        else:
            raise TypeError(
                f"The gate {gate.name} is not valid when converting to qtrl string."
            )
        qtrl_strs.append(qtrl_str)
    return qtrl_strs


def qiskit_circuit_to_qasm_string(circ: QuantumCircuit) -> str:
    """Converts a circuit to a QASM string.
    
    This function is required to transpile circuits that contain C0C0iX and CCZ gates. 
    The ``QuantumCircuit.qasm()`` method in Qiskit does not implement these gates.
    
    Args:
        circ: The circuit to be transformed to a QASM string.
    
    Returns:
        qasm_str: The QASM string of the circuit.
    """
    # The header of the QASM string.
    qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

    # Define 3q gates c0c0ix and ccz, since these are not defined in the standard library.
    qasm_str += (
        "gate c0c0ix p0,p1,p2 {x p0; x p1; ccx p0,p1,p2; cp(pi/2) p0,p1; x p0; x p1;}\n"
    )
    qasm_str += "gate ccz p0,p1,p2 {h p2; ccx p0,p1,p2; h p2;}\n"

    # Define quantum and classical registers.
    if len(circ.qregs) > 0:
        assert len(circ.qregs[0]) == 1
        n_qubits = len(circ.qregs[0])
        qasm_str += f"qreg q[{n_qubits}];\n"
    if len(circ.cregs) > 0:
        assert len(circ.cregs) == 1
        n_clbits = len(circ.cregs[0])
        qasm_str += f"creg c[{n_clbits}];\n"

    for inst, qargs, cargs in circ.data:
        # Build instruction string, quantum register string and
        # optionally classical register string.
        if len(inst.params) > 0 and not isinstance(inst.params[0], np.ndarray):
            # 1q or 2q gate with parameters
            params_str = ",".join([str(x) for x in inst.params])
            inst_str = f"{inst.name}({params_str})"
        else:  # 1q, 2q gate without parameters, 3q gate, measure or barrier
            inst_str = inst.name
        qargs_inds = [q._index for q in qargs]
        qargs_str = ",".join([f"q[{i}]" for i in qargs_inds])
        if cargs != []:
            cargs_inds = [c._index for c in cargs]
            cargs_str = ",".join([f"c[{i}]" for i in cargs_inds])

        # Only measure requires a `inst qargs -> cargs` format. All the other gates
        # follow the same `inst qargs` format.
        if inst.name in [
            "rz",
            "rx",
            "ry",
            "h",
            "x",
            "p",
            "u3",
            "cz",
            "swap",
            "cp",
            "barrier",
            "c0c0ix",
            "ccz",
        ]:
            qasm_str += f"{inst_str} {qargs_str};\n"
        elif inst.name == "measure":
            qasm_str += f"{inst_str} {qargs_str} -> {cargs_str};\n"
        else:
            raise TypeError(
                f"Instruction {inst.name} cannot be converted to QASM string."
            )

    # Temporary check statement.
    circ_new = QuantumCircuit.from_qasm_str(qasm_str)
    assert circuit_equal(circ, circ_new)
    return qasm_str


def convert_circuit_to_string(circ: QuantumCircuit, kind: str) -> str:
    """Converts a circuit to its qtrl or qasm string form.
    
    Args:
        circ: A circuit to be converted to a string.
        kind: The type of output string, ``"qtrl"`` or ``"qasm"``.
    
    Returns:
        string: The string created from the circuit.
    """
    assert kind in ["qtrl", "qasm"]
    if kind == "qtrl":
        string = qiskit_circuit_to_qtrl_strings(circ)
        string = json.dumps(string)
    else:
        string = qiskit_circuit_to_qasm_string(circ)
    return string


def convert_string_to_circuit(string: str, kind: str) -> QuantumCircuit:
    """Converts a qtrl or qasm string to its circuit form.
    
    Args:
        string: A qtrl or qasm string to be converted to a circuit.
        kind: The type of input string, ``"qtrl"`` or ``"qasm"``.
    
    Returns:
        circ: The circuit created from the string.
    """
    assert kind in ["qtrl", "qasm"]
    if kind == "qtrl":
        qtrl_strs = json.loads(string)
        circ = qtrl_strings_to_qiskit_circuit(qtrl_strs)
    else:
        qasm_str = string.decode()
        circ = QuantumCircuit.from_qasm_str(qasm_str)
    return circ
