"""
===============================================================
Ground State Solver (:mod:`fd_greens.main.ground_state_solver`)
===============================================================
"""

from typing import Optional, Sequence, Tuple
import h5py
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, Aer
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance

from .molecular_hamiltonian import MolecularHamiltonian
from .ansatze import AnsatzFunction, build_ansatz_gs
from .z2symmetries import transform_4q_pauli
from ..utils import get_statevector, write_hdf5, vqe_minimize


class GroundStateSolver:
    """A class to solve the ground state energy and state."""

    def __init__(
        self,
        h: MolecularHamiltonian,
        ansatz_func: AnsatzFunction = build_ansatz_gs,
        init_params: Sequence[float] = [1, 2, 3, 4],
        q_instance: Optional[QuantumInstance] = None,
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
        self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=[1, 1])
        self.ansatz_func = ansatz_func
        self.init_params = init_params
        if q_instance is None:
            self.q_instance = QuantumInstance(Aer.get_backend("statevector_simulator"))
        else:
            self.q_instance = q_instance
        self.method = method
        self.h5fname = h5fname + ".h5"

        self.energy = None
        self.ansatz = None

    def run_exact(self) -> None:
        """Calculates the exact ground state of the Hamiltonian using 
        exact 2q gate decomposition."""
        from qiskit.quantum_info import TwoQubitBasisDecomposer
        from qiskit.extensions import CZGate

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
        decomp = TwoQubitBasisDecomposer(CZGate())
        self.ansatz = decomp(U)

        print(f"Ground state energy is {self.energy:.3f} eV")

    def run_vqe(self) -> None:
        """Calculates the ground state of the Hamiltonian using VQE."""
        assert self.ansatz_func is not None
        assert self.q_instance is not None

        # if self.load_params:
        #     print("Load VQE circuit from file")
        #     with open('data/vqe_energy.txt', 'r') as f:
        #         self.energy = float(f.read())
        #     with open('circuits/vqe_circuit.txt') as f:
        #         self.ansatz = QuantumCircuit.from_qasm_str(f.read())
        # else:

        # from qiskit.extensions import UnitaryGate
        # from qiskit.quantum_info import TwoQubitBasisDecomposer
        # from qiskit.extensions import CZGate
        # from utils import create_circuit_from_inst_tups

        self.energy, self.ansatz = vqe_minimize(
            self.h_op, self.ansatz_func, self.init_params, self.q_instance
        )

        # v0 = [ 0.68777917+0.j,  0.07105691+0.j, -0.07105691+0.j, -0.71893091+0.j]
        # U = np.array([v0, [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
        # U = np.linalg.qr(U)[0]
        # decomp = TwoQubitBasisDecomposer(CZGate())
        # self.ansatz = decomp(U)

        # from utils import get_statevector
        # print('psi in gs solver =', get_statevector(self.ansatz))

        print(f"Ground state energy is {self.energy:.3f} eV")

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
