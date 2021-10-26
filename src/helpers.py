import numpy as np

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import Barrier
from qiskit.extensions import UnitaryGate, SwapGate

from hamiltonians import MolecularHamiltonian

def get_lih_hamiltonian(r: float) -> MolecularHamiltonian:
    """Returns the HOMO-LUMO LiH Hamiltonian with bond length r."""
    hamiltonian = MolecularHamiltonian(
        [['Li', (0, 0, 0)], ['H', (0, 0, r)]], 'sto3g', 
        occ_inds=[0], act_inds=[1, 2])
    return hamiltonian

def get_quantum_instance(type_str) -> QuantumInstance:
    """Returns the QuantumInstance from type string."""
    if type_str == 'sv':
        q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
    elif type_str == 'qasm':
        q_instance = QuantumInstance(Aer.get_backend('qasm_simulator', shots=10000))
    elif type_str == 'noisy':
        q_instance = QuantumInstance(Aer.get_backend('qasm_simulator', shots=10000, noise_model_name='ibmq_jakarta'))
    return q_instance

def get_berkeley_ccx_data():
    iX = np.array([[0, 1j], [1j, 0]])
    ccx_data = [(SwapGate(), [1, 2]), 
                (Barrier(4), [0, 1, 2, 3]), 
                (UnitaryGate(iX).control(2), [0, 2, 1]), 
                (Barrier(4), [0, 1, 2, 3]),
                (SwapGate(), [1, 2])]
    return ccx_data