"""Circuit utility module."""

from typing import Tuple, Iterable, Union, List, Optional, Sequence
import numpy as np
import h5py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance
from qiskit.circuit import Instruction, Qubit, Clbit
from qiskit.opflow import PauliSumOp


QubitLike = Union[int, Qubit]
ClbitLike = Union[int, Clbit]
InstructionTuple = Tuple[Instruction, List[QubitLike], Optional[List[ClbitLike]]]
QuantumCircuitLike = Union[QuantumCircuit, Iterable[InstructionTuple]]


def remove_instructions_in_circuit(
    circ_like: QuantumCircuitLike, instructions: Iterable[str]
) -> QuantumCircuitLike:
    """Removes certain instructions in a circuit.
    
    Args:
        circ_like: The circuit or instruction tuples on which certain instructions are removed.
        instructions: An iterable of strings representing instruction names.
        
    Returns:
        A new quantum circuit on which certain instructions are removed.
    """
    if isinstance(circ_like, QuantumCircuit):
        inst_tups = circ_like.data.copy()
    else:
        inst_tups = circ_like
    qreg, creg = get_registers_in_inst_tups(inst_tups)
    # qreg = circ.qregs[0] if circ.qregs != [] else None
    # creg = circ.cregs[0] if circ.cregs != [] else None

    inst_tups_new = []
    for inst_tup in inst_tups:
        if inst_tup[0].name not in instructions:
            inst_tups_new.append(inst_tup)

    if isinstance(circ_like, QuantumCircuit):
        circ_new = create_circuit_from_inst_tups(inst_tups_new, qreg=qreg, creg=creg)
        return circ_new
    else:
        return inst_tups_new


remove_instructions = remove_instructions_in_circuit


def get_registers_in_circuit(
    circ_like: QuantumCircuitLike,
) -> Tuple[Optional[QuantumRegister], Optional[ClassicalRegister]]:
    """Returns the quantum and classical registers from a quantum circuit. 
    
    Qubits and classical bits can be specified either as Qubit/Clbit instances or as 
    integers in the circuit. New instances of QuantumRegister and ClassicalRegister will
    be created with name 'q' and 'c'. If the quantum or classical register is found, it is 
    returned as None.

    Args:
        The instruction tuples from which qreg and creg are extracted.

    Returns:
        qreg: The quantum register in the instruction tuples.
        creg: The classical register in the instruction tuples.
    """
    # Extract the instruction tuples from the quantum circuit.
    if isinstance(circ_like, QuantumCircuit):
        inst_tups = circ_like.data.copy()
    else:
        inst_tups = circ_like

    # Find the number of qubits and number of classical bits in the circuit.
    n_qubits = 0
    n_clbits = 0
    for inst_tup in inst_tups:
        _, qargs, cargs = inst_tup
        if qargs != []:
            # Take the largest index in the qargs.
            if isinstance(qargs[0], int):  # int
                max_q = max(qargs)
            else:  # Qubit
                max_q = max([q._index for q in qargs])
            # If max qubit index is larger than n_qubits, update n_qubits.
            if max_q + 1 > n_qubits:
                n_qubits = max_q + 1
        if cargs != []:
            # Take the largest index in the cargs.
            if isinstance(cargs[0], int):  # int
                max_c = max(cargs)
            else:  # Qubit
                max_c = max([c._index for c in cargs])
            # If max classical bit index is larger than n_clbits, update n_clbits.
            if max_c + 1 > n_clbits:
                n_clbits = max_c + 1

    # Create new qreg with number of qubits and creg with number of classical bits
    # as in the original circuit.
    if n_qubits > 0:
        qreg = QuantumRegister(n_qubits, name="q")
    else:
        qreg = None
    if n_clbits > 0:
        creg = ClassicalRegister(n_clbits, name="c")
    else:
        creg = None
    return qreg, creg


get_registers_in_inst_tups = get_registers_in_circuit


def create_circuit_from_inst_tups(
    inst_tups: Iterable[InstructionTuple],
    qreg: Optional[QuantumRegister] = None,
    creg: Optional[ClassicalRegister] = None,
) -> QuantumCircuit:
    """Creates a circuit from instruction tuples.

    The arguments in qargs and cargs should be specified as integers. If specified as Qubit/Clbits 
    bound to specific QuantumRegister/ClassicalRegisters, either both qreg and creg are specified, 
    or the new qargs and cargs are bound to new QuantumRegister/ClassicalRegisters with names 'q'
    and 'c'.
    
    Args:
        inst_tups: Instruction tuples from which the circuit is to be constructed.
        qreg: The quantum register in the circuit. If not passed in,
            will be a new QuantumRegister with name 'q'.
        creg: The classical register in the circuit. If not passed in,
            will be a new ClassicalRegister with name 'c'.
        
    Returns:
    	A quantum circuit constructed from the instruction tuples.
    """
    # Obtain the quantum register and classical register in the instruction tuples,
    # and create a QuantumCircuit with them.
    if qreg is None and creg is None:
        qreg, creg = get_registers_in_inst_tups(inst_tups)
    regs = [reg for reg in [qreg, creg] if reg is not None]
    circ = QuantumCircuit(*regs)

    for inst, qargs, cargs in inst_tups:
        # If arguments of qargs and cargs are specified as Qubit/Clbits,
        # convert them to integers since they are not neccesarily the same
        # as the qreg and creg extracted using get_registers_in_inst_tups.
        try:
            qargs = [q._index for q in qargs]
        except:
            pass
        circ.append(inst, qargs, cargs)
    return circ


def split_circuit_across_barriers(circ: QuantumCircuit) -> List[List[InstructionTuple]]:
    """Splits a circuit into instruction tuples across barriers."""
    inst_tups = circ.data.copy()
    inst_tups_all = []  # all inst_tups_single
    inst_tups_single = []  # temporary variable to hold inst_tups_all components

    # Split when encoutering a barrier
    for i, inst_tup in enumerate(inst_tups):
        if inst_tup[0].name == "barrier":  # append and start a new inst_tups_single
            inst_tups_all.append(inst_tups_single)
            inst_tups_single = []
        elif i == len(inst_tups) - 1:  # append and stop
            inst_tups_single.append(inst_tup)
            inst_tups_all.append(inst_tups_single)
        else:  # just append
            inst_tups_single.append(inst_tup)

    return inst_tups_all


def measure_operator(
    circ: QuantumCircuit,
    op: PauliSumOp,
    q_instance: QuantumInstance,
    anc_state: Sequence[int] = [],
    qreg: Optional[QuantumRegister] = None,
) -> float:
    """Measures an operator on a circuit.

    Args:
        circ: The quantum circuit to be measured.
        op: The operator to be measured.
        q_instance: The QuantumInstance on which to execute the circuit.
        anc_state: A sequence of integers indicating the ancilla states.
        qreg: The quantum register on which the operator is measured.
    
    Returns:
        The value of the operator on the circuit.
    """
    if qreg is None:
        qreg = circ.qregs[0]
    n_qubits = len(qreg)
    n_anc = len(anc_state)
    n_sys = n_qubits - n_anc

    circ_list = []
    for term in op:
        label = term.primitive.table.to_labels()[0]

        # Create circuit for measuring this Pauli string
        circ_meas = circ.copy()
        creg = ClassicalRegister(n_qubits)
        circ_meas.add_register(creg)

        for i in range(n_anc):
            circ_meas.measure(i, i)

        for i in range(n_sys):
            if label[n_sys - 1 - i] == "X":
                circ_meas.h(qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == "Y":
                circ_meas.rx(np.pi / 2, qreg[i + n_anc])
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])
            elif label[n_sys - 1 - i] == "Z":
                circ_meas.measure(qreg[i + n_anc], creg[i + n_anc])

        circ_list.append(circ_meas)

    result = q_instance.execute(circ_list)
    counts_list = result.get_counts()

    value = 0
    for i, counts in enumerate(counts_list):
        # print('counts =', counts)
        coeff = op.primitive.coeffs[i]
        counts_new = counts.copy()

        for key, val in counts.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if key_anc != anc_state:
                del counts_new[key]

        shots = sum(counts_new.values())
        for key, val in counts_new.items():
            key_list = [int(c) for c in list(reversed(key))]
            key_anc = key_list[:n_anc]
            key_sys = key_list[n_anc:]
            if sum(key_sys) % 2 == 0:
                value += coeff * val / shots
            else:
                value -= coeff * val / shots
    return value


def get_circuit_depth(h5fname: str, circ_label: str) -> int:
    """Returns the circuit depth of a circuit saved in an HDF5 file."""
    h5file = h5py.File(h5fname + ".h5", "r")
    dset = h5file[f"circ{circ_label}/transpiled"]
    circ = QuantumCircuit.from_qasm_str(dset[()].decode())
    depth = len(circ)
    return depth


def get_n_2q_gates(h5fname: str, circ_label: str) -> int:
    """Returns the number of 2q gates in a circuit saved in an HDF5 file."""
    h5file = h5py.File(h5fname + ".h5", "r")
    dset = h5file[f"circ{circ_label}/transpiled"]
    qasm_str = dset[()].decode()
    circ = QuantumCircuit.from_qasm_str(qasm_str)
    count = 0
    for _, qargs, _ in circ.data:
        if len(qargs) == 2:
            count += 1
    return count


def get_n_3q_gates(h5fname: str, circ_label: str) -> int:
    """Returns the number of 3q gates in a circuit saved in an HDF5 file."""
    h5file = h5py.File(h5fname + ".h5", "r")
    dset = h5file[f"circ{circ_label}/transpiled"]
    qasm_str = dset[()].decode()
    circ = QuantumCircuit.from_qasm_str(qasm_str)
    count = 0
    for _, qargs, _ in circ.data:
        if len(qargs) == 3:
            count += 1
    return count
