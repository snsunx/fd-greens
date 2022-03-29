"""
==============================================================
Tomography Utilities (:mod:`fd_greens.utils.tomography_utils`)
==============================================================
"""


from typing import Optional, Tuple, Iterable, Union
import itertools
import numpy as np
from permutation import Permutation

from qiskit import Aer, QuantumCircuit, ClassicalRegister
from qiskit.utils import QuantumInstance
from qiskit.circuit import Barrier, Measure, Qubit
from qiskit.extensions import RXGate, RZGate
from qiskit.ignis.verification.tomography import (
    state_tomography_circuits,
    StateTomographyFitter,
)

from .circuit_utils import create_circuit_from_inst_tups

QubitLike = Union[int, Qubit]

# Basis matrix for tomography
basis_matrix = []
bases = list(itertools.product("xyz", "xyz", "01", "01"))
states = {
    "x0": np.array([1.0, 1.0]) / np.sqrt(2),
    "x1": np.array([1.0, -1.0]) / np.sqrt(2),
    "y0": np.array([1.0, 1.0j]) / np.sqrt(2),
    "y1": np.array([1.0, -1.0j]) / np.sqrt(2),
    "z0": np.array([1.0, 0.0]),
    "z1": np.array([0.0, 1.0]),
}

for basis in bases:
    label0 = "".join([basis[0], basis[3]])
    label1 = "".join([basis[1], basis[2]])
    state0 = states[label0]
    state1 = states[label1]
    state = np.kron(state1, state0)
    rho_vec = np.outer(state, state.conj()).reshape(-1)
    basis_matrix.append(rho_vec)

basis_matrix = np.array(basis_matrix)


def state_tomography(
    circ: QuantumCircuit, q_instance: Optional[QuantumInstance] = None
) -> np.ndarray:
    """Performs state tomography on a quantum circuit.

    Args:
        circ: The quantum circuit to perform tomography on.
        q_instance: The QuantumInstance to execute the circuit.

    Returns:
        The density matrix obtained from state tomography.
    """
    if q_instance is None:
        backend = Aer.get_backend("qasm_simulator")
        q_instance = QuantumInstance(backend, shots=8192)

    qreg = circ.qregs[0]
    qst_circs = state_tomography_circuits(circ, qreg)
    if True:
        fig = qst_circs[5].draw(output="mpl")
        fig.savefig("qst_circ.png")
    result = q_instance.execute(qst_circs)
    qst_fitter = StateTomographyFitter(result, qst_circs)
    rho_fit = qst_fitter.fit(method="lstsq")
    return rho_fit


def append_tomography_gates(
    circ: QuantumCircuit, qubits: Iterable[QubitLike], label: Tuple[str]
) -> QuantumCircuit:
    """Appends tomography gates to a circuit.

    Args:
        circ: The circuit to which tomography gates are to be appended.
        qubits: The qubits to be tomographed.
        label: The tomography states label.
    
    Returns:
        A new circuit with tomography gates appended.
    """
    assert len(qubits) == len(label)
    inst_tups = circ.data.copy()
    inst_tups_swap = []
    perms = []

    # Split off the last few SWAP gates.
    while True:
        inst, qargs, cargs = inst_tups.pop()
        if inst.name == "swap":
            inst_tups_swap.insert(0, (inst, qargs, cargs))
            qinds = [q._index for q in qargs]
            perms.append(Permutation.cycle(*[i + 1 for i in qinds]))
        else:
            inst_tups.append((inst, qargs, cargs))
            break

    # Append rotation gates when tomographing on X or Y.
    for q, s in zip(qubits, label):
        q_new = q + 1
        for perm in perms:
            q_new = perm(q_new)
        q_new -= 1
        if s == "x":
            inst_tups += [
                (RZGate(np.pi / 2), [q_new], []),
                (RXGate(np.pi / 2), [q_new], []),
                (RZGate(np.pi / 2), [q_new], []),
            ]
        elif s == "y":
            inst_tups += [
                (RXGate(np.pi / 2), [q_new], []),
                (RZGate(np.pi / 2), [q_new], []),
            ]

    inst_tups += inst_tups_swap
    tomo_circ = create_circuit_from_inst_tups(inst_tups)
    return tomo_circ


def append_measurement_gates(circ: QuantumCircuit) -> QuantumCircuit:
    """Appends measurement gates to a circuit.
    
    Args:
        The circuit to which measurement gates are to be appended.
        
    Returns:
        A new circuit with measurement gates appended.
    """
    qreg = circ.qregs[0]
    n_qubits = len(qreg)
    circ.add_register(ClassicalRegister(n_qubits))
    inst_tups = circ.data.copy()
    # if n_qubits == 4:
    #     perms = [Permutation.cycle(2, 3)]
    # else:
    perms = []

    # Split off the last few SWAP gates.
    while True:
        inst, qargs, cargs = inst_tups.pop()
        if inst.name == "swap":
            qinds = [q._index for q in qargs]
            perms.append(Permutation.cycle(*[i + 1 for i in qinds]))
        else:
            inst_tups.append((inst, qargs, cargs))
            break

    # Append the measurement gates permuted by the SWAP gates.
    inst_tups += [(Barrier(n_qubits), qreg, [])]
    for c in range(n_qubits):
        q = c + 1
        for perm in perms:
            q = perm(q)
        q -= 1

        inst_tups += [(Measure(), [q], [c])]
    # circ.measure(range(n_qubits), range(n_qubits))

    circ_new = create_circuit_from_inst_tups(inst_tups)
    return circ_new
