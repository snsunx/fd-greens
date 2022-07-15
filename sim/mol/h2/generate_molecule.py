import sys
sys.path.append('../../..')

import numpy as np
import matplotlib.pyplot as plt

from fd_greens import get_h2_hamiltonian, MethodIndicesPairs

def main():

    bond_distances = np.arange(0.2, 4.0, 0.1)
    energies_gs = []
    energies_es = []
    gs_state_probs0 = []
    gs_state_probs1 = []
    gs_state_probs2 = []
    gs_state_probs3 = []

    for r in bond_distances:
        hamiltonian = get_h2_hamiltonian(r)
        hamiltonian.transform(MethodIndicesPairs.get_pairs('d'))
        matrix = hamiltonian.matrix
        e, v = np.linalg.eigh(matrix)
        v0 = v[:, 0]

        energies_gs.append(e[0])
        energies_es.append(e[1])

        gs_state_probs0.append(abs(v0[0]) ** 2)
        gs_state_probs1.append(abs(v0[1]) ** 2)
        gs_state_probs2.append(abs(v0[2]) ** 2)
        gs_state_probs3.append(abs(v0[3]) ** 2)

    fig, ax = plt.subplots()
    ax.plot(bond_distances, energies_gs, marker='o', label="Ground state")
    ax.plot(bond_distances, energies_es, marker='o', label="1st excited state")
    ax.legend()
    fig.savefig('h2_energies.png', dpi=250, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(bond_distances, gs_state_probs0, marker='o', label="00")
    ax.plot(bond_distances, gs_state_probs1, marker='x', label="01")
    ax.plot(bond_distances, gs_state_probs2, marker='3', label="10")
    ax.plot(bond_distances, gs_state_probs3, marker='4', label="11")
    ax.legend()
    fig.savefig('h2_states.png', dpi=250, bbox_inches='tight')


    

if __name__ == '__main__':
    main()
