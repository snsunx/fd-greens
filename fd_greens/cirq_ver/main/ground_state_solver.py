"""
===============================================================
Ground State Solver (:mod:`fd_greens.main.ground_state_solver`)
===============================================================
"""

from typing import Sequence, List

import numpy as np
import h5py
import cirq

from .molecular_hamiltonian import MolecularHamiltonian
from .z2symmetries import transform_4q_pauli
from ..utils import write_hdf5, decompose_1q_gate, unitary_equal


class GroundStateSolver:
    """Ground state solver."""

    def __init__(
        self, h: MolecularHamiltonian, qubits: Sequence[cirq.Qid], h5fname: str = "lih"
    ) -> None:
        """Initializes a ``GroudStateSolver`` object.
        
        Args:
            h: The molecular Hamiltonian object.
            qubits: The qubits on which the ground state is prepared.
            h5fname: The hdf5 file name.
        """
        self.h = h
        self.qubits = qubits
        self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=[1, 1])
        self.h5fname = h5fname + ".h5"

        self.energy = None
        self.ansatz = None

    def _run_exact(self) -> None:
        """Calculates the ground state using exact two-qubit gate decomposition."""
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
        # print("M conj T\n", cirq.unitary(cirq.Circuit(operations_M_conjugate)))
        circuit = cirq.Circuit(all_operations)

        assert unitary_equal(cirq.unitary(circuit), U)
        self.ansatz = circuit
        print(f"Ground state energy is {self.energy:.3f} eV")

    def _save_data(self) -> None:
        """Saves ground state energy and ground state ansatz to hdf5 file."""
        h5file = h5py.File(self.h5fname, "r+")

        write_hdf5(h5file, "gs", "energy", self.energy)
        write_hdf5(h5file, "gs", "ansatz", self.ansatz.qasm())

        h5file.close()

    def run(self) -> None:
        """Runs the ground state calculation."""
        self._run_exact()
        self._save_data()


def get_AB_operations(
    U: np.ndarray, qubits: Sequence[cirq.Qid]
) -> List[cirq.Operation]:
    """Returns the operations of :math:`A\otimes B` in :math:`O(4)` gate decomposition.

    This function uses Theorem 2 of Vatan and Williams, Phys. Rev. A 69, 032315 (2004) to 
    decomposes a two-qubit unitary in :math:`O(4)` into a tensor product of two single-qubit
    unitaries. 
    
    Args:
        U: The unitary to be decomposed.
        qubits: The qubits on which the unitary acts on.
        
    Returns:
        operations: The operations of :math:`A\otimes B` in the unitary decomposition.
    """
    assert U.shape == (4, 4)
    assert len(qubits) == 2

    M = np.array(
        [[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]
    ) / np.sqrt(2)
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
    """Returns the operations of the magic basis matrix.

    The magic basis matrix decomposition is given in Fig. 1 of Vatan and Williams, 
    Phys. Rev. A 69, 032315 (2004), which is

    .. math::
    
        (H\otimes I) CZ (H\otimes H) (S\otimes S)
    
    Args:
        qubits: The qubits on which the magic basis matrix acts.
        swapped: Whether the two qubits are swapped.
        conjugate: Whether to take the complex conjugate of the matrix.
    
    Returns:
        operations: The operations of the magic basis matrix.
    """
    operations = [
        cirq.rz(np.pi / 2)(qubits[0]),
        cirq.rz(np.pi / 2)(qubits[1]),
        cirq.H(qubits[0]),
        cirq.H(qubits[1]),
        # cirq.rz(np.pi / 2)(qubits[0]),
        # cirq.rx(np.pi / 2)(qubits[0]),
        # cirq.rz(np.pi / 2)(qubits[0]),
        # cirq.rz(np.pi / 2)(qubits[1]),
        # cirq.rx(np.pi / 2)(qubits[1]),
        # cirq.rz(np.pi / 2)(qubits[1]),
        cirq.CZ(qubits[0], qubits[1]),
    ]
    if not swapped:
        operations += [
            # cirq.rz(np.pi / 2)(qubits[0]),
            # cirq.rx(np.pi / 2)(qubits[0]),
            # cirq.rz(np.pi / 2)(qubits[0]),
            cirq.H(qubits[0])
        ]
    else:
        operations += [
            # cirq.rz(np.pi / 2)(qubits[1]),
            # cirq.rx(np.pi / 2)(qubits[1]),
            # cirq.rz(np.pi / 2)(qubits[1]),
            cirq.H(qubits[1])
        ]

    if conjugate:
        operations = [cirq.inverse(op) for op in operations[::-1]]

    return operations
