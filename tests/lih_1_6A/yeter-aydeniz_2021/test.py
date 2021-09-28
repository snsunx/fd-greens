import sys
sys.path.append('../../src/')
import numpy as np
from vqe import *

rs = np.arange(0.5, 3.3, 0.1)
es_ne1_min = []
es_ne1_max = []
es_ne3_min = []
es_ne3_max = []
es_gs = []

for r in rs:
    print('r =', r)
    qubit_operator = build_qubit_operator(
        'Li 0 0 0; H 0 0 ' + str(r), 
        occupied_indices=[0], active_indices=[1, 2])
    """
    ansatz = build_ne1_ansatz(4)
    e, _ = run_vqe(ansatz.copy(), qubit_operator)
    es_ne1_min.append(e)
    e, _ = run_vqe(ansatz.copy(), qubit_operator, mode='max')
    es_ne1_max.append(e)

    ansatz = build_ne3_ansatz(4)
    e, _ = run_vqe(ansatz.copy(), qubit_operator)
    es_ne3_min.append(e)
    e, _ = run_vqe(ansatz.copy(), qubit_operator, mode='max')
    es_ne3_max.append(e)
    """
    
    ansatz = build_ne2_ansatz(4)
    #print(ansatz)
    e, _ = run_vqe(ansatz.copy(), qubit_operator)
    print('e =', e)
    es_gs.append(e)

#np.savetxt('es_ne1_min.dat', np.vstack((rs, es_ne1_min)).T)
#np.savetxt('es_ne1_max.dat', np.vstack((rs, es_ne1_max)).T)
#np.savetxt('es_ne3_min.dat', np.vstack((rs, es_ne3_min)).T)
#np.savetxt('es_ne3_max.dat', np.vstack((rs, es_ne3_max)).T)
#np.savetxt('es_gs.dat', np.vstack((rs, es_gs)).T)
