import sys
sys.path.append('../../src')
import numpy as np
from vqe import *
from ansatze import *
from circuits import *

qubit_op = build_qubit_operator(
    'Li 0 0 0; H 0 0 3', 
    occupied_indices=[0], active_indices=[1, 2])
ansatz = build_ne2_ansatz(4)
e, ansatz = run_vqe(ansatz.copy(), qubit_op)


def exact(ind_left, ind_right):
    circ = get_off_diagonal_circuits(ansatz, ind_left, ind_right, measure=False)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend, shots=8000)
    result = job.result()
    psi = result.get_statevector()
    probs = [0, 0]
    for i in range(64):
        b = bin(i)[2:]
        b = [int(d) for d in b]
        b = [0] * (6 - len(b)) + b
        if sum(b[:4]) == 1:
            probs[0] += abs(psi[i]) ** 2
        elif sum(b[:4]) == 3:
            probs[1] += abs(psi[i]) ** 2
    return probs

def sampling(ind_left, ind_right):
    circ = get_off_diagonal_circuits(ansatz, ind_left, ind_right)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend, shots=8000)
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

#for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
for i, j in [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)]:
    probs_exact = exact(i, j)
    probs_sampling = sampling(i, j)
    print('{} {}, {:.6f}'.format(i, j, abs(probs_exact[0] - probs_sampling[0])))
exact(0, 1)
sampling(0, 1)

