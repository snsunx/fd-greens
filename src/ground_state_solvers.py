"""Ground state solver module."""

from typing import Optional, Sequence, Tuple
import h5py
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance

from hamiltonians import MolecularHamiltonian
from ansatze import AnsatzFunction, build_ansatz_gs
from operators import transform_4q_pauli
from utils import create_circuit_from_inst_tups, get_lih_hamiltonian, get_quantum_instance, get_statevector, write_hdf5


class GroundStateSolver:
    """A class to solve the ground state energy and state."""

    def __init__(self,
                 h: MolecularHamiltonian, 
                 ansatz_func: AnsatzFunction = build_ansatz_gs,
                 init_params: Sequence[float] = [1, 2, 3, 4],
                 q_instance: QuantumInstance = get_quantum_instance('sv'),
                 method: str = 'exact',
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
        self.method = method
        self.h5fname = h5fname + '.h5'

        self.energy = None
        self.ansatz = None

    def run_exact(self) -> None:
        """Calculates the exact ground state of the Hamiltonian. Same as
        running VQE with statevector simulator."""
        # self.q_instance = get_quantum_instance('sv')
        # self.run_vqe()

        from qiskit.quantum_info import TwoQubitBasisDecomposer
        from qiskit.extensions import CZGate

        e, v = np.linalg.eigh(self.h_op.to_matrix())
        self.energy = e[0]
        # v0 = [ 0.6877791696238387+0j, 0.07105690514886635+0j, 0.07105690514886635+0j, -0.7189309050895454+0j]
        v0 = v[:, 0][abs(v[:, 0])>1e-8]
        U = np.array([v0, [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
        U = np.linalg.qr(U)[0]
        decomp = TwoQubitBasisDecomposer(CZGate())
        self.ansatz = decomp(U)

        print(f'Ground state energy is {self.energy:.3f} eV')
    
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
            self.h_op, self.ansatz_func, self.init_params, self.q_instance)
        
        # v0 = [ 0.68777917+0.j,  0.07105691+0.j, -0.07105691+0.j, -0.71893091+0.j]
        # U = np.array([v0, [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).T
        # U = np.linalg.qr(U)[0]
        # decomp = TwoQubitBasisDecomposer(CZGate())
        # self.ansatz = decomp(U)

        # from utils import get_statevector
        # print('psi in gs solver =', get_statevector(self.ansatz))
        

        print(f'Ground state energy is {self.energy:.3f} eV')

    def save_data(self) -> None:
        """Saves ground state energy and ground state ansatz to hdf5 file."""
        h5file = h5py.File(self.h5fname, 'r+')

        write_hdf5(h5file, 'gs', 'energy', self.energy)
        write_hdf5(h5file, 'gs', 'ansatz', self.ansatz.qasm())
        
        h5file.close()

    def run(self, method: Optional[str] = None) -> None:
        """Runs the ground state calculation."""
        if method is not None: self.method = method
        if self.method == 'exact': self.run_exact()
        elif self.method == 'vqe': self.run_vqe()
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
        print("STATEVECTOR")
        obj_func = obj_func_sv
    else:
        obj_func = obj_func_qasm
    res = minimize(obj_func, x0=init_params, method='L-BFGS-B')
    energy = res.fun
    ansatz = ansatz_func(res.x)
    print('-' * 80)
    print('in vqe_minimize')
    print('energy =', energy)
    print('ansatz =', ansatz)
    for inst, qargs, cargs in ansatz.data:
        print(inst.name, inst.params, qargs)
    with np.printoptions(precision=15):
        psi = get_statevector(ansatz)
        print(psi)

    h = get_lih_hamiltonian(5.0)
    h_arr = transform_4q_pauli(h.qiskit_op, init_state=[1, 1]).to_matrix()
    print(psi.conj().T @ h_arr @ psi)

    e, v = np.linalg.eigh(h_arr)
    print(e[0])
    print(v[:, 0])
    print('-' * 80)

    return energy, ansatz