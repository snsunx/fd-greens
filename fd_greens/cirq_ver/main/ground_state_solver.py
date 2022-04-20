"""
===============================================================
Ground State Solver (:mod:`fd_greens.main.ground_state_solver`)
===============================================================
"""

from typing import Optional, Sequence, Tuple, List
import h5py
import numpy as np
from scipy.optimize import minimize

import cirq
from qiskit import QuantumCircuit, Aer
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance

from .molecular_hamiltonian import MolecularHamiltonian
from .ansatze import AnsatzFunction, build_ansatz_gs
from .z2symmetries import transform_4q_pauli
from ..utils import (
    get_statevector,
    write_hdf5,
    vqe_minimize,
    decompose_1q_gate,
    unitary_equal,
)


class GroundStateSolver:
    """A class to solve the ground state energy and state."""

    def __init__(
        self,
        h: MolecularHamiltonian,
        qubits: Sequence[cirq.Qid],
        # ansatz_func: AnsatzFunction = build_ansatz_gs,
        # init_params: Sequence[float] = [1, 2, 3, 4],
        method: str = "exact",
        h5fname: str = "lih",
    ) -> None:
        """Initializes a ``GroudStateSolver`` object.
        
        Args:
            h: The molecular Hamiltonian object.
            ansatz_func: The ansatz function for VQE.
            init_params: Initial guess parameters for VQE.
            q_instance: The quantum instance for VQE.
            h5fname: The hdf5 file name.
            dsetname: The dataset name in the hdf5 file.
        """
        self.h = h
        self.qubits = qubits
        self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=[1, 1])
        # self.ansatz_func = ansatz_func
        # self.init_params = init_params
        self.method = method
        self.h5fname = h5fname + ".h5"

        self.energy = None
        self.ansatz = None

    def run_exact(self) -> None:
        """Calculates the exact ground state of the Hamiltonian using 
        exact 2q gate decomposition."""
        # Solve for the eigenvalue and eigenvectors of the Hamiltonian. The energy is the smallest
        # eigenvalue. The ansatz is prepared by the unitary with the lowest eigenvector as the first
        # column and the rest filled in randomly. The matrix after going through a QR factorization
        # and a KAK decomposition is assigned as the ansatz.
        e, v = np.linalg.eigh(self.h_op.to_matrix())
        self.energy = e[0]
        # v0 = [ 0.6877791696238387+0j, 0.07105690514886635+0j, 0.07105690514886635+0j, -0.7189309050895454+0j]
        v0 = v[:, 0][abs(v[:, 0]) > 1e-8]
        U = np.array([v0, [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
        U = np.linalg.qr(U)[0]

        operations_AB = get_AB_operations(U, self.qubits)
        operations_M_swapped = get_magic_basis_operations(self.qubits, swapped=True)
        operations_M_conjugate = get_magic_basis_operations(self.qubits, conjugate=True)

        all_operations = (
            operations_M_swapped
            + [cirq.rz(np.pi)(self.qubits[0])]
            + operations_AB
            + operations_M_conjugate
        )
        print("M conj T\n", cirq.unitary(cirq.Circuit(operations_M_conjugate)))
        circuit = cirq.Circuit(all_operations)

        assert unitary_equal(cirq.unitary(circuit), U)
        self.ansatz = circuit
        print(f"Ground state energy is {self.energy:.3f} eV")

    def run_vqe(self) -> None:
        """Calculates the ground state of the Hamiltonian using VQE."""
        raise NotImplementedError("VQE for ground state is not implemented.")

    def save_data(self) -> None:
        """Saves ground state energy and ground state ansatz to hdf5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        write_hdf5(h5file, "gs", "energy", self.energy)
        write_hdf5(h5file, "gs", "ansatz", self.ansatz.qasm())

        h5file.close()

    def run(self, method: Optional[str] = None) -> None:
        """Runs the ground state calculation."""
        if method is not None:
            self.method = method
        if self.method == "exact":
            self.run_exact()
        elif self.method == "vqe":
            self.run_vqe()
        self.save_data()


def get_AB_operations(
    U: np.ndarray, qubits: Sequence[cirq.Qid]
) -> Tuple[List[cirq.Operation], List[cirq.Operation]]:
    """Decomposes a two-qubit unitary into a tensor product of two unitaries.
    
    Args:
        U: The unitary to be decomposed.
        qubits: The qubits on which the unitary acts on.
        
    Returns:
        circuit_A: The circuit of the unitary acting on the first qubit.
        circuit_B: The circuit of the unitary acting on the second qubit.
    """
    assert U.shape == (4, 4)
    assert len(qubits) == 2

    M = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
    IZ = np.kron(np.eye(2), cirq.unitary(cirq.Z))
    SWAP = cirq.unitary(cirq.SWAP)
    U1 = M @ U @ M.conj().T @ IZ @ SWAP
    
    U_reshaped = np.array([])

    for i in range(2):
        for j in range(2):
            U_sub = U1[2 * i : 2 * (i + 1)][:, 2 * j : 2 * (j + 1)]
            U_sub_flat = U_sub.flatten()
            U_reshaped = np.hstack((U_reshaped, U_sub_flat))

    U_reshaped = U_reshaped.reshape(4, 4)
    u, s, vd = np.linalg.svd(U_reshaped)
    print(s)
    assert np.count_nonzero(np.abs(s) > 1e-8) == 1

    A = u[:, 0].reshape(2, 2) * np.sqrt(s[0])
    B = vd[0].reshape(2, 2) * np.sqrt(s[0])
    assert unitary_equal(np.kron(A, B), U1)

    operations_A = decompose_1q_gate(A, qubits[0])
    operations_B = decompose_1q_gate(B, qubits[1])
    operations = operations_A + operations_B
    return operations


def get_magic_basis_operations(
    qubits: Sequence[cirq.Qid], swapped: bool = False, conjugate: bool = False
) -> cirq.Circuit:
    """Returns the circuit corresponding to the magic basis."""
    operations = [
        cirq.rz(np.pi / 2)(qubits[0]),
        cirq.rz(np.pi / 2)(qubits[1]),
        cirq.rz(np.pi / 2)(qubits[0]),
        cirq.rx(np.pi / 2)(qubits[0]),
        cirq.rz(np.pi / 2)(qubits[0]),
        cirq.rz(np.pi / 2)(qubits[1]),
        cirq.rx(np.pi / 2)(qubits[1]),
        cirq.rz(np.pi / 2)(qubits[1]),
        cirq.CZ(qubits[0], qubits[1]),
    ]
    if not swapped:
        operations += [
            cirq.rz(np.pi / 2)(qubits[0]),
            cirq.rx(np.pi / 2)(qubits[0]),
            cirq.rz(np.pi / 2)(qubits[0]),
        ]
    else:
        operations += [
            cirq.rz(np.pi / 2)(qubits[1]),
            cirq.rx(np.pi / 2)(qubits[1]),
            cirq.rz(np.pi / 2)(qubits[1]),
        ]

    if conjugate:
        operations = [cirq.inverse(op) for op in operations[::-1]]

    return operations
