import sys

sys.path.append("../src/")
from vqe import vqe_minimize
from hamiltonians import MolecularHamiltonian
from z2_symmetries import transform_4q_hamiltonian
from constants import HARTREE_TO_EV

from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.utils import QuantumInstance


def main():
    hamiltonian = MolecularHamiltonian(
        [["Li", (0, 0, 0)], ["H", (0, 0, 3.0)]], "sto3g", occ_inds=[0], act_inds=[1, 2]
    )

    qiskit_op = transform_4q_hamiltonian(hamiltonian.qiskit_op, init_state=[1, 1])
    qiskit_op = qiskit_op.reduce()

    q_instance = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=5000)
    energy, ansatz = vqe_minimize(qiskit_op, q_instance)

    print("energy =", energy * HARTREE_TO_EV, "eV")
    print(ansatz)


if __name__ == "__main__":
    main()
