from typing import Optional, Sequence, Tuple, Callable

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance

AnsatzFunction = Callable[[Sequence[float]], QuantumCircuit]

def get_ansatz(params: Sequence[float]) -> QuantumCircuit:
    """Constructs an ansatz for a two-qubit Hamiltonian."""
    qreg = QuantumRegister(2, name='q')
    ansatz = QuantumCircuit(qreg)
    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)
    ansatz.cz(0, 1)
    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)
    return ansatz

def get_ansatz_e(params: float) -> QuantumCircuit:
    """Returns the ansatz for (N+1)-electron states."""
    qreg = QuantumRegister(2, name='q')
    ansatz = QuantumCircuit(qreg)
    ansatz.ry(params[0], 0)
    ansatz.x(1)
    return ansatz

def get_ansatz_h(params: Sequence[float]) -> QuantumCircuit:
    """Returns the ansatz for (N-1)-electron states."""
    qreg = QuantumRegister(2, name='q')
    ansatz = QuantumCircuit(qreg)
    ansatz.ry(params[0], 0)
    return ansatz

def objective_function_sv(params: Sequence[float],
                          hamiltonian_op: PauliSumOp,
                          ansatz_func: Optional[AnsatzFunction] = None,
                          q_instance: Optional[QuantumInstance] = None) -> float:
    """VQE objective function for ground state without sampling."""
    ansatz = ansatz_func(params).copy()
    result = q_instance.execute(ansatz)
    statevector = result.get_statevector()

    matrix = hamiltonian_op.to_matrix()

    energy = statevector.conj().T @ matrix @ statevector
    return energy.real


def objective_function_qasm(params: Sequence[float], 
                            hamiltonian_op: PauliSumOp,
                            ansatz_func: Optional[AnsatzFunction] = None,
                            q_instance: Optional[QuantumInstance] = None) -> float:
    """VQE objective function for ground state with sampling."""
    shots = q_instance.run_config.shots

    label_coeff_list = hamiltonian_op.primitive.to_list()
    label_coeff_dict = dict(label_coeff_list)

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

def vqe_minimize(hamiltonian_op: PauliSumOp,
                 ansatz_func: Optional[AnsatzFunction] = None,
                 init_params: Optional[Sequence[float]] = None,
                 q_instance: Optional[QuantumInstance] = None
                 ) -> Tuple[float, QuantumCircuit]:
    """Minimizes the energy of a Hamiltonian using VQE."""
    if q_instance.backend.name() == 'statevector_simulator':
        objective_function = objective_function_sv
    else:
        objective_function = objective_function_qasm

    if ansatz_func is None:
        ansatz_func = get_ansatz
    if init_params is None:
        init_params = [-5., 0., 0., 5.]

    #res = minimize(objective_function, x0=[-5., 0., 0., 5.],
    #               method='Powell', args=(hamiltonian_op, q_instance))
    res = minimize(objective_function, x0=init_params, method='Powell',
                   args=(hamiltonian_op, ansatz_func, q_instance))
    energy = res.fun
    ansatz = ansatz_func(res.x)
    return energy, ansatz
