"""Quantum Phase Estimation."""

import numpy as np
from qiskit import *

def qpe(circ, n=1):
    """Iterative Quantum Phase Estimation."""
    assert len(circ.qregs) == 2
    qreg_anc = circ.qregs[0]
    qreg_sys = circ.qregs[1]    
    assert len(qreg_anc) == 1
    assert len(qreg_sys) == 1
    creg = circ.cregs[0]
    
    digits = []
    
    for i in range(n):
        qreg_anc = QuantumRegister(1)
        qreg_sys = QuantumRegister(1)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg_anc, qreg_sys, creg)
        backend = Aer.get_backend('qasm_simulator')

        circ.h(qreg_anc)
        circ.x(qreg_sys)
        
        circ.cp(theta * 2 ** (n - 1 - i), qreg_anc, qreg_sys)
        circ.barrier()
        for j, d in enumerate(digits):
            circ.p(-2 * np.pi * d / 2 ** (j + 2), qreg_anc[0])
        circ.h(qreg_anc)
        circ.measure(qreg_anc, creg)
        print(circ)
            
        job = execute(circ, backend=backend)
        result = job.result()
        counts = result.get_counts()
        d_new = max(counts, key=counts.get)
        
        digits.insert(0, int(d_new))
        
    return digits
