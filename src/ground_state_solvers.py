"""Ground state solver module."""

from typing import Sequence, Tuple
import h5py
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance

from hamiltonians import MolecularHamiltonian
from ansatze import AnsatzFunction, build_ansatz_gs
from operators import transform_4q_pauli
from utils import get_quantum_instance

class GroundStateSolver:
    """A class to solve the ground state energy and state."""

    def __init__(self,
                 h: MolecularHamiltonian, 
                 ansatz_func: AnsatzFunction = build_ansatz_gs,
                 init_params: Sequence[float] = [-5., 0., 0., 5.],
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 h5fname: str = 'lih') -> None:
        """Initializes a GroudStateSolver object.
        
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
        self.q_instance = q_instance
        self.h5fname = h5fname + '.h5'

        self.energy = None
        self.ansatz = None

    def run_exact(self) -> None:
        """Calculates the exact ground state of the Hamiltonian. Same as
        running VQE with statevector simulator."""
        self.q_instance = get_quantum_instance('sv')
        self.run_vqe()
    
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
        self.energy, self.ansatz = vqe_minimize(
            self.h_op, self.ansatz_func, self.init_params, self.q_instance)

        print(f'Ground state energy is {self.energy:.3f} eV')

    def save_data(self) -> None:
        """Saves ground state energy and ground state ansatz to hdf5 file."""
        h5file = h5py.File(self.h5fname, 'r+')

        h5file['gs/energy'] = self.energy
        h5file['gs/ansatz'] = self.ansatz.qasm()
        
        h5file.close()

    def run(self, method: str = 'vqe') -> None:
        """Runs the ground state calculation."""
        if method == 'exact': self.run_exact()
        elif method == 'vqe': self.run_vqe()
        self.save_data()

def vqe_minimize(h_op: PauliSumOp,
                 ansatz_func: AnsatzFunction,
                 init_params: Sequence[float],
                 q_instance: QuantumInstance
                 ) -> Tuple[float, QuantumCircuit]:
    """Minimizes the energy of a Hamiltonian using VQE.
    
    Args:
        h_op: The Hamiltonian operator.
        ansatz_func: The ansatz function for VQE.
        init_params: Initial guess parameters.
        q_instance: The quantum instance for executing the circuits.
    
    Returns:
        energy: The VQE ground state energy.
        ansatz: The VQE ground state ansatz.
    """

    def obj_func_sv(params):
        """VQE objective function for statevector simulator."""
        ansatz = ansatz_func(params).copy()
        result = q_instance.execute(ansatz)
        psi = result.get_statevector()
        energy = psi.conj().T @ h_op.to_matrix() @ psi
        return energy.real

    def obj_func_qasm(params):
        """VQE objective function for QASM simulator and hardware."""
        # This function assumes the Hamiltonian only has the terms 
        # II, IZ, ZI, ZZ, IX, ZX, XI, XZ, YY.
        shots = q_instance.run_config.shots

        label_coeff_list = h_op.primitive.to_list()
        label_coeff_dict = dict(label_coeff_list)
        for key in ['II', 'IZ', 'ZI', 'ZZ', 'IX', 'ZX', 'XI', 'XZ', 'YY']:
            if key not in label_coeff_dict.keys():
                label_coeff_dict[key] = 0.

        energy = label_coeff_dict['II']

        # Measure in ZZ basis
        ansatz = ansatz_func(params).copy()
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ['00', '01', '10', '11']:
            if key not in counts.keys():
                counts[key] = 0
        energy += label_coeff_dict['IZ'] * (counts['00'] + counts['10'] - counts['01'] - counts['11']) / shots
        energy += label_coeff_dict['ZI'] * (counts['00'] + counts['01'] - counts['10'] - counts['11']) / shots
        energy += label_coeff_dict['ZZ'] * (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots

        # Measure in ZX basis
        ansatz = ansatz_func(params).copy()
        ansatz.h(0)
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ['00', '01', '10', '11']:
            if key not in counts.keys():
                counts[key] = 0
        energy += label_coeff_dict['IX'] * (counts['00'] + counts['10'] - counts['01'] - counts['11']) / shots
        energy += label_coeff_dict['ZX'] * (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots

        # Measure in XZ basis
        ansatz = ansatz_func(params).copy()
        ansatz.h(1)
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ['00', '01', '10', '11']:
            if key not in counts.keys():
                counts[key] = 0
        energy += label_coeff_dict['XI'] * (counts['00'] + counts['01'] - counts['10'] - counts['11']) / shots
        energy += label_coeff_dict['XZ'] * (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots

        # Measure in YY basis
        ansatz = ansatz_func(params).copy()
        ansatz.rx(np.pi / 2, 0)
        ansatz.rx(np.pi / 2, 1)
        ansatz.measure_all()
        result = q_instance.execute(ansatz)
        counts = result.get_counts()
        for key in ['00', '01', '10', '11']:
            if key not in counts.keys():
                counts[key] = 0
        energy += label_coeff_dict['YY'] * (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots

        energy = energy.real
        return energy

    if q_instance.backend.name() == 'statevector_simulator':
        obj_func = obj_func_sv
    else:
        obj_func = obj_func_qasm
    res = minimize(obj_func, x0=init_params, method='Powell')
    energy = res.fun
    ansatz = ansatz_func(res.x)
    return energy, ansatz