import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.utils import QuantumInstance

def get_ansatz(params):
    """Constructs an ansatz for a two-qubit Hamiltonian."""
    qreg = QuantumRegister(2, name='q')
    ansatz = QuantumCircuit(qreg)
    ansatz.ry(params[0], 0)
    ansatz.ry(params[1], 1)
    ansatz.cz(0, 1)
    ansatz.ry(params[2], 0)
    ansatz.ry(params[3], 1)
    return ansatz

def objective_function_gs(params, hamiltonian_op, q_instance=None):
    """VQE objective function for ground state."""
    shots = q_instance.run_config.shots

    label_coeff_list = hamiltonian_op.primitive.to_list()
    label_coeff_dict = dict(label_coeff_list)

    energy = label_coeff_dict['II']

    # Measure in ZZ basis
    ansatz = get_ansatz(params).copy()
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
    ansatz = get_ansatz(params).copy()
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
    ansatz = get_ansatz(params).copy()
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
    ansatz = get_ansatz(params).copy()
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

def vqe_minimize(hamiltonian_op, q_instance):
    """Minimize the energy of a Hamiltonian using VQE."""
    res = minimize(objective_function_gs, x0=[-5., 0., 0., 5.], 
                   method='Powell', args=(hamiltonian_op, q_instance))
    
    energy = res.fun
    ansatz = get_ansatz(res.x)
    return energy, ansatz