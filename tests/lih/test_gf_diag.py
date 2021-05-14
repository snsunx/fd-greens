import sys
sys.path.append('../../src')
import numpy as np
from vqe import *
from ansatze import *
from circuits import *

qubit_op = build_qubit_operator(
    'Li 0 0 0; H 0 0 3', 
    occupied_indices=[0], active_indices=[1, 2])
mat = qubit_op.to_matrix()
proj_ne1 = np.zeros((16, 4))
proj_ne1[int('0001', 2), 0] = 1.
proj_ne1[int('0010', 2), 1] = 1.
proj_ne1[int('0100', 2), 2] = 1.
proj_ne1[int('1000', 2), 3] = 1.
mat_ne1 = proj_ne1.T @ mat @ proj_ne1
e_ne1, v_ne1 = np.linalg.eigh(mat_ne1)
print(e_ne1)

proj_ne3 = np.zeros((16, 4))
proj_ne3[int('0111', 2), 0] = 1.
proj_ne3[int('1011', 2), 1] = 1.
proj_ne3[int('1101', 2), 2] = 1.
proj_ne3[int('1110', 2), 3] = 1.
mat_ne3 = proj_ne3.T @ mat @ proj_ne3
e_ne3, v_ne3 = np.linalg.eigh(mat_ne3)
print(e_ne3)

ansatz = build_ne2_ansatz(4)
e, ansatz = run_vqe(ansatz.copy(), qubit_op)


def exact(ind):
    diag_circ = get_diagonal_circuits(ansatz, ind, measure=False)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(diag_circ, backend)
    result = job.result()
    psi = result.get_statevector()
    probs = [0, 0]
    for i in range(32):
        b = bin(i)[2:]
        b = [int(d) for d in b]
        b = [0] * (5 - len(b)) + b
        if sum(b[:-1]) == 1:
            probs[0] += abs(psi[i]) ** 2
        elif sum(b[:-1]) == 3:
            probs[1] += abs(psi[i]) ** 2
    return probs

def sampling(ind):
    diag_circ = get_diagonal_circuits(ansatz, ind)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(diag_circ, backend, shots=8000)
    result = job.result()
    counts = result.get_counts()
    probs = [None, None]
    for key, val in counts.items():
        if key[-1] == '0':
            probs[0] = val
        elif key[-1] == '1':
            probs[1] = val
    probs = [p / sum(probs) for p in probs]
    return probs

for i in range(4):
    probs_exact = exact(0)
    probs_sampling = sampling(0)
    print('{}, {:.6f}'.format(i, abs(probs_exact[0] - probs_sampling[0])))
