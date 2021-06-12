from qiskit import *
from math import pi
from numpy import sin, cos

theta = 3 / 2
print(cos(theta / 2) ** 2)
print(sin(theta / 2) ** 2)

circ = QuantumCircuit(2, 1)
circ.h(0)
circ.rx(theta, 1)
circ.cz(0, 1)
circ.h(0)
circ.measure(0, 0)

backend = Aer.get_backend('qasm_simulator')
job = execute(circ, backend=backend)
result = job.result()
counts = result.get_counts()
shots = sum(counts.values())
for key, val in counts.items():
    print(key, val / shots)
