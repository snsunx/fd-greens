"""
===============================================================
Ground State Solver (:mod:`fd_greens.main.ground_state_solver`)
===============================================================
"""

from typing import Sequence, List

import numpy as np
import h5py
import cirq
import json
from qiskit.quantum_info import OneQubitEulerDecomposer

from .molecular_hamiltonian import MolecularHamiltonian
from .circuit_string_converter import CircuitStringConverter
from .transpilation import transpile_into_berkeley_gates
from .parameters import method_indices_pairs
from .utilities import unitary_equal


class GroundStateSolver:
    """Ground state solver."""

    def __init__(
        self, hamiltonian: MolecularHamiltonian, qubits: Sequence[cirq.Qid], h5fname: str = "lih"
    ) -> None:
        """Initializes a ``GroudStateSolver`` object.
        
        Args:
            hamiltonian: The molecular Hamiltonian object.
            qubits: The qubits on which the ground state is prepared.
            h5fname: The hdf5 file name.
        """
        self.hamiltonian = hamiltonian
        self.hamiltonian.transform(method_indices_pairs)

        self.qubits = qubits        
        self.h5fname = h5fname + ".h5"
        self.circuit_string_converter = CircuitStringConverter(qubits)

    def _run_exact(self) -> None:
        """Calculates the ground state using exact two-qubit gate decomposition."""
        # Solve for the eigenvalue and eigenvectors of the Hamiltonian. The energy is the smallest
        # eigenvalue. The ansatz is prepared by the unitary with the lowest eigenvector as the first
        # column and the rest filled in randomly. The matrix after going through a QR factorization
        # and a KAK decomposition is assigned as the ansatz.
        e, v = np.linalg.eigh(self.hamiltonian.matrix)
        self.energy = e[0]
        # v0 = [ 0.6877791696238387+0j, 0.07105690514886635+0j, 0.07105690514886635+0j, -0.7189309050895454+0j]
        v0 = v[:, 0][abs(v[:, 0]) > 1e-8]
        U = np.array([v0, [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
        U = np.linalg.qr(U)[0]

        operations_AB = self._get_AB_operations(U)
        operations_M_swapped = self._get_magic_basis_operations(swapped=True)
        operations_M_conjugate = self._get_magic_basis_operations(conjugate=True)

        all_operations = (
            operations_M_swapped
            + [cirq.rz(np.pi)(self.qubits[0])]
            + operations_AB
            + operations_M_conjugate
        )
        # print("M conj T\n", cirq.unitary(cirq.Circuit(operations_M_conjugate)))
        circuit = cirq.Circuit(all_operations)
        circuit = transpile_into_berkeley_gates(circuit)

        assert unitary_equal(cirq.unitary(circuit), U)
        self.ansatz = circuit
        print(f"Ground state energy is {self.energy:.3f} eV")

    def _get_AB_operations(self, U: np.ndarray) -> List[cirq.Operation]:
        """Returns the operations of :math:`A\otimes B` in :math:`O(4)` gate decomposition.

        This function uses Theorem 2 of Vatan and Williams, Phys. Rev. A 69, 032315 (2004) to 
        decomposes a two-qubit unitary in :math:`O(4)` into a tensor product of two single-qubit
        unitaries. 
        """
        assert U.shape == (4, 4)

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
        assert np.count_nonzero(np.abs(s) > 1e-8) == 1

        A = u[:, 0].reshape(2, 2) * np.sqrt(s[0])
        B = vd[0].reshape(2, 2) * np.sqrt(s[0])
        assert unitary_equal(np.kron(A, B), U1)

        operations_A = self._decompose_1q_gate(A, self.qubits[0])
        operations_B = self._decompose_1q_gate(B, self.qubits[1])
        operations = operations_A + operations_B
        return operations

    def _get_magic_basis_operations(
        self, swapped: bool = False, conjugate: bool = False
    ) -> cirq.Circuit:
        """Returns the operations of the magic basis matrix.

        The magic basis matrix decomposition is given in Fig. 1 of Vatan and Williams, 
        Phys. Rev. A 69, 032315 (2004), which is

        .. math::
        
            (H\otimes I) CZ (H\otimes H) (S\otimes S)
        """
        operations = [
            cirq.S(self.qubits[0]),
            cirq.S(self.qubits[1]),
            cirq.H(self.qubits[0]),
            cirq.H(self.qubits[1]),
            cirq.CZ(self.qubits[0], self.qubits[1]),
        ]

        if not swapped:
            operations += [cirq.H(self.qubits[0])]
        else:
            operations += [cirq.H(self.qubits[1])]

        if conjugate:
            operations = [cirq.inverse(op) for op in operations[::-1]]

        return operations

    # TODO: This should be handled differently.
    def _decompose_1q_gate(self, unitary: np.ndarray, qubit: cirq.Qid) -> List[cirq.Operation]:
        """ZXZXZ decomposition of a single-qubit gate."""
        decomposer = OneQubitEulerDecomposer("U3")
        unitary_decomposed = decomposer(unitary)
        theta, phi, lam = unitary_decomposed[0][0].params

        operations = [
            cirq.rz(lam - np.pi)(qubit),
            cirq.rx(np.pi / 2)(qubit),
            cirq.rz(np.pi - theta)(qubit),
            cirq.rx(np.pi / 2)(qubit),
            cirq.rz(phi)(qubit),
        ]

        assert unitary_equal(cirq.Circuit(operations).unitary(), unitary)
        return operations

    def _save_data(self) -> None:
        """Saves ground state energy and ansatz to hdf5 file."""
        h5file = h5py.File(self.h5fname, "r+")
        qtrl_strings = self.circuit_string_converter.convert_circuit_to_strings(self.ansatz)

        for dset_name in ['gs/energy', 'gs/ansatz']:
            if dset_name in h5file:
                del h5file[dset_name]
        
        h5file['gs/energy'] = self.energy
        h5file['gs/ansatz'] = json.dumps(qtrl_strings)

        h5file.close()

    def run(self) -> None:
        """Runs the ground-state calculation with exact two-qubit gate decomposition."""
        self._run_exact()
        self._save_data()
