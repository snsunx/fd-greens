from typing import Optional, Sequence, Tuple, Callable, Union

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance

from hamiltonians import MolecularHamiltonian
from utils import save_circuit
from ansatze import AnsatzFunction, build_ansatz_gs
from z2_symmetries import transform_4q_pauli

def objective_function_sv(params: Sequence[float],
                          h_op: PauliSumOp,
                          ansatz_func: Optional[AnsatzFunction] = None,
                          q_instance: Optional[QuantumInstance] = None) -> float:
    """VQE objective function for ground state without sampling."""
    ansatz = ansatz_func(params).copy()
    result = q_instance.execute(ansatz)
    statevector = result.get_statevector()

    matrix = h_op.to_matrix()

    energy = statevector.conj().T @ matrix @ statevector
    return energy.real


def objective_function_qasm(params: Sequence[float], 
                            h_op: PauliSumOp,
                            ansatz_func: Optional[AnsatzFunction] = None,
                            q_instance: Optional[QuantumInstance] = None) -> float:
    """VQE objective function with sampling."""
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

def vqe_minimize(h_op: PauliSumOp,
                 ansatz_func: Optional[AnsatzFunction] = None,
                 init_params: Optional[Sequence[float]] = None,
                 q_instance: Optional[QuantumInstance] = None
                 ) -> Tuple[float, QuantumCircuit]:
    """Minimizes the energy of a Hamiltonian using VQE.
    
    Args:
        h_op: The Hamiltonian operator.
        ansatz_func: An ansatz function that takes in parameters and returns a quantum circuit.
        init_params: Initial guess parameters.
        q_instance: The QuantumInstance for executing the circuits.
    
    Returns:
        energy: The VQE ground state energy.
        ansatz: The VQE ground state ansatz.
    """
    if q_instance.backend.name() == 'statevector_simulator':
        objective_function = objective_function_sv
    else:
        objective_function = objective_function_qasm

    if ansatz_func is None:
        ansatz_func = build_ansatz_gs
    if init_params is None:
        init_params = [-5., 0., 0., 5.]

    res = minimize(objective_function, x0=init_params, method='Powell',
                   args=(h_op, ansatz_func, q_instance))
    energy = res.fun
    ansatz = ansatz_func(res.x)
    return energy, ansatz

class GroundStateSolver:
    def __init__(self,
                 h: MolecularHamiltonian, 
                 ansatz_func: Optional[AnsatzFunction] = None,
                 init_params: Optional[Sequence[float]] = None,
                 q_instance: Optional[QuantumInstance] = None,
                 save_params: bool = False,
                 load_params: bool = False):
        """Initializes a VQE object.
        
        Args:
            h_op: The Hamiltonian operator.
            ansatz_func: The ansatz function for VQE.
            init_params: Initial guess parameters for VQE.
            q_instance: The QuantumInstance for VQE.
            save_params: Whether to save parameters.
            load_params: Whether to load parameters.
        """
        self.h = h
        self.h_op = transform_4q_pauli(self.h.qiskit_op, init_state=[1, 1])
        
        self.ansatz_func = ansatz_func
        self.init_params = init_params
        self.q_instance = q_instance
        self.save_params = save_params
        self.load_params = load_params

        self.state = None
        self.ansatz = None
    
    def _run_exact(self):
        """Calculates the exact ground state of the Hamiltonian."""
        # e, v = np.linalg.eigh(self.h_op.to_matrix())
        # self.energy = e[0]
        # self.state = v[:, 0]
        # print(f'Ground state energy = {self.energy:.3f} eV')
        from helpers import get_quantum_instance
        self.q_instance = get_quantum_instance('sv')
        self._run_vqe()

    
    def _run_vqe(self):
        """Calculates the ground state of the Hamiltonian using VQE."""
        assert self.ansatz_func is not None
        assert self.q_instance is not None

        if self.load_params:
            print("Load VQE circuit from file")
            with open('data/vqe_energy.txt', 'r') as f: 
                self.energy = float(f.read())
            with open('circuits/vqe_circuit.txt') as f:
                self.ansatz = QuantumCircuit.from_qasm_str(f.read())
        else:
            # print("===== Start calculating the ground state using VQE =====")
            self.energy, self.ansatz = vqe_minimize(
                self.h_op, ansatz_func=self.ansatz_func, 
                init_params=self.init_params, q_instance=self.q_instance)
            # print("===== Finish calculating the ground state using VQE =====")

            if self.save_params:
                print("Save VQE circuit to file")
                with open('data/vqe_energy.txt', 'w') as f: 
                    f.write(str(self.energy))
                save_circuit(self.ansatz.copy(), 'circuits/vqe_circuit')

        print(f'Ground state energy = {self.energy:.3f} eV')

    def run(self, method='vqe'):
        """Runs the N+/-1 electron states calculation."""
        if method == 'exact':
            self._run_exact()
        elif method == 'vqe':
            self._run_vqe()