"""Helper functions module."""

from typing import Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, IBMQ
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from ..main.molecular_hamiltonian import MolecularHamiltonian


def get_lih_hamiltonian(r: float) -> MolecularHamiltonian:
    """Returns the HOMO-LUMO LiH Hamiltonian with bond length r.
    
    Args:
        The bond length of the molecule in Angstrom.
    
    Returns:
        The molecular Hamiltonian.
    """
    hamiltonian = MolecularHamiltonian(
        [["Li", (0, 0, 0)], ["H", (0, 0, r)]], "sto3g", occ_inds=[0], act_inds=[1, 2]
    )
    return hamiltonian


def get_quantum_instance(type_str: str) -> QuantumInstance:
    """Returns the QuantumInstance based on the input string.
    
    Three types of quantum instances are returned from this function: the
    statevector simulator ('sv'), the QASM simulator ('qasm'), and the noisy
    simulator ('noisy').

    Args:
        A string indicating the type of the quantum instance.
    
    Returns:
        The pre-defined quantum instance based on the input string.
    """
    if type_str == "sv":
        q_instance = QuantumInstance(Aer.get_backend("statevector_simulator"))
    elif type_str == "qasm":
        q_instance = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=10000)
    elif type_str == "noisy":
        IBMQ.load_account()
        provider = IBMQ.get_provider(
            hub="ibm-q-research", group="caltech-1", project="main"
        )
        backend_sim = Aer.get_backend("qasm_simulator")
        backend = provider.get_backend("ibmq_jakarta")
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
        q_instance = QuantumInstance(
            backend_sim,
            noise_model=noise_model,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            shots=10000,
        )
    return q_instance
