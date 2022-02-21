import numpy as np
import matplotlib.pyplot as plt

def main_A():
    omegas, lih_expt_qasm = np.loadtxt('data/lih_3A_1_A.dat').T

    fig, ax = plt.subplots()
    ax.plot(omegas, lih_expt_qasm, ls='--', marker='x', markevery=5, label='QASM')

    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
    #ax.set_xlim([-20, 10])
    ax.legend()
    fig.savefig('figs/A.png', dpi=300)

def main_TrS():
    omegas, lih_expt_qasm, _ = np.loadtxt(f'data/{h5fname}{suffix}_TrS.dat').T

    fig, ax = plt.subplots()
    ax.plot(omegas, lih_expt_qasm, ls='--', marker='x', markevery=20, label='QASM')

    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel("Tr$\Sigma$ (eV)")
    ax.set_xlim([-20, 10])
    ax.legend()
    fig.savefig('figs/TrSigma.png', dpi=300)

if __name__ == '__main__':
    h5fname = 'lih_3A'
    suffix = ''

    # main_A()
    # main_TrS()
