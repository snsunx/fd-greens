from typing import Tuple, Optional, Iterable
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from openfermion.ops import PolynomialTensor
from openfermion.transforms import get_fermion_operator, jordan_wigner
from tools import get_pauli_tuple

# Change Term to PauliTuple
PauliTuple = Tuple[Tuple[str, int]]

def build_diagonal_circuits(ansatz: QuantumCircuit, 
                            tup_xy: Iterable[PauliTuple],
                            with_qpe: bool = True,
                            add_barriers: bool = True,
                            measure: bool = False) -> QuantumCircuit:
    """Constructs the circuit to calculate a diagonal transition amplitude.
    
    Args:
        ansatz: The ansatz quantum circuit corresponding to the ground state.
        tup_xy: The creation/annihilation operator of the circuit in tuple form.
        with_qpe: Whether an additional qubit is added for QPE.
        add_barriers: Whether barriers are added to the circuit.
        measure: Whether the ancilla qubits are measured.

    Returns:
        A QuantumCircuit with the diagonal Pauli string appended.
    """
    # Create a new circuit along with the quantum registers
    n_qubits = ansatz.num_qubits
    n_anc = 2 if with_qpe else 1
    qreg = QuantumRegister(n_qubits + n_anc)
    creg = ClassicalRegister(n_anc)
    circ = QuantumCircuit(qreg, creg)

    # Copy instructions from the ansatz circuit to the new circuit
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + n_anc for qubit in qargs]
        circ.append(inst, qargs, cargs)

    # Apply the gates corresponding to the creation/annihilation terms
    if add_barriers:
        circ.barrier()
    circ.h(0)
    if add_barriers:
        circ.barrier()
    apply_cU(circ, tup_xy[0], ctrl=0, offset=n_anc)
    if add_barriers:
        circ.barrier()
    apply_cU(circ, tup_xy[1], ctrl=1, offset=n_anc)
    if add_barriers:
        circ.barrier()
    circ.h(0)
    if add_barriers:
        circ.barrier()

    if measure:
        circ.measure(0, 0)
    
    return circ

def build_off_diagonal_circuits(ansatz: QuantumCircuit,
                                tup_xy_left: Iterable[PauliTuple],
                                tup_xy_right: Iterable[PauliTuple],
                                with_qpe: bool = True,
                                add_barriers: bool = True,
                                measure: bool = False) -> QuantumCircuit:
    """Constructs the circuit to calculate off-diagonal transition amplitudes.
    
    Args:
        ansatz: The ansatz quantum circuit corresponding to the ground state.
        tup_xy_left: The left creation/annihilation operator of the circuit 
            in tuple form.
        tup_xy_right: The right creation/annihilation operator of the circuit
            in tuple form.
        with_qpe: Whether an additional qubit is added for QPE.
        add_barriers: Whether barriers are added to the circuit.
        measure: Whether the ancilla qubits are measured.

    Returns:
        A QuantumCircuit with the off-diagonal Pauli string appended.
    """
    # Create a new circuit along with the quantum registers
    n_qubits = ansatz.num_qubits
    n_anc = 3 if with_qpe else 2
    qreg = QuantumRegister(n_qubits + n_anc)
    creg = ClassicalRegister(n_anc)
    circ = QuantumCircuit(qreg, creg)

    # Copy instructions from the ansatz circuit to the new circuit
    for inst, qargs, cargs in ansatz.data:
        qargs = [qubit._index + n_anc for qubit in qargs]
        circ.append(inst, qargs, cargs)

    # Apply the gates corresponding to the creation/annihilation terms
    if add_barriers:
        circ.barrier()
    circ.h([0, 1])
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, tup_xy_left[0], ctrl=(0, 0), offset=n_anc)
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, tup_xy_left[1], ctrl=(1, 0), offset=n_anc)
    if add_barriers:
        circ.barrier()
    circ.rz(np.pi / 4, 1)
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, tup_xy_right[0], ctrl=(0, 1), offset=n_anc)
    if add_barriers:
        circ.barrier()
    apply_ccU(circ, tup_xy_right[1], ctrl=(1, 1), offset=n_anc)
    if add_barriers:
        circ.barrier()
    circ.h([0, 1])

    if measure:
        circ.measure([0, 1], [0, 1])

    return circ

def apply_cU(circ: QuantumCircuit, 
             term: Tuple[Tuple[int, str]], 
             ctrl: int = 1, 
             offset: int = 1) -> None:
    """Applies a controlled-U gate to a quantum circuit.
    
    Args:
        circ: The quantum circuit to which the controlled U gate is appended.
        term: A tuple specifying the Pauli string corresponding to 
            the creation/annihilation operator, e.g. Z0Z1X2 is specified as 
            (('Z', 0), ('Z', 1), ('X', 2)).
        ctrl: An integer indicating the qubit state on which the controlled-U
            gate is controlled on. Must be 0 or 1.
        offset: An integer indicating the number of qubits skipped when 
            applying the controlled-U gate.
    """
    if term == ():
        return
        
    assert ctrl in [0, 1]
    ind_max = max([t[0] for t in term])

    # Prepend X gates for control on 0
    if ctrl == 0:
        circ.x(0)
    
    # Prepend gates for Pauli
    if term[-1][1] == 'X':
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.s(0)
        circ.rx(np.pi / 2, ind_max + offset)
    
    # Implement multi-qubit gate
    for i in range(ind_max + offset, offset, -1):
        circ.cx(i, i - 1)
    circ.cz(0, offset)
    for i in range(offset, ind_max + offset):
        circ.cx(i + 1, i)
    
    # Append gates for Pauli
    if term[-1][1] == 'X':
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.rx(-np.pi / 2, ind_max + offset)
    
    # Append X gates for control on 0
    if ctrl == 0:
        circ.x(0)

def apply_ccU(circ: QuantumCircuit, 
              term: Tuple[Tuple[str, int]], 
              ctrl: Tuple[int, int] = (1, 1), 
              offset: int = 2) -> None:
    """Applies a controlled-controlled-U gate to a quantum circuit.
    
    Args:
        circ: The quantum circuit to which the controlled-U gate is appended.
        term: A tuple specifying the Pauli string corresponding to 
            the creation/annihilation operator, e.g. Z0Z1X2 is specified as 
            (('Z', 0), ('Z', 1), ('X', 2)).
        ctrl: An tuple of two integers indicating the qubit states on which 
            the controlled-controlled-U gate is controlled on. Both integers 
            must be 0 or 1.
        offset: An integer indicating the number of qubits skipped when 
            applying the controlled-controlled-U gate.    
    """
    if term == ():
        return

    assert set(ctrl).issubset({0, 1})
    ind_max = max([t[0] for t in term])
    
    # Prepend X gates when controlled on 0
    if ctrl[0] == 0:
        circ.x(0)
    if ctrl[1] == 0:
        circ.x(1)

    #if ctrl[0] == 1:
    #    circ.s(0)

    # Prepend rotation gates in the case of X, Y
    if term[-1][1] == 'X':
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.cp(np.pi / 2, 0, 1)
        circ.rx(np.pi / 2, ind_max + offset)
    
    # Implement multi-qubit gate
    for i in range(ind_max + offset, offset, -1):
        circ.cx(i, i - 1)
    circ.h(offset)
    circ.ccx(0, 1, offset)
    circ.h(offset)
    for i in range(offset, ind_max + offset):
        circ.cx(i + 1, i)
    
    # Append rotation gates in the case of X, Y
    if term[-1][1] == 'X':
        circ.h(ind_max + offset)
    elif term[-1][1] == 'Y':
        circ.rx(-np.pi / 2, ind_max + offset)
    
    # Append X gates when controlled on 0
    if ctrl[0] == 0:
        circ.x(0)
    if ctrl[1] == 0:
        circ.x(1)
