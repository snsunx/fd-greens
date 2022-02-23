import numpy as np
import matplotlib.pyplot as plt

def main_A():
    omegas, As1 = np.loadtxt('data/lih_3A1_u_A.dat').T
    omegas, As2 = np.loadtxt('data/lih_3A1_d_A.dat').T

    fig, ax = plt.subplots()
    ax.plot(omegas, As1, ls='--', marker='x', markevery=5, label='u')
    ax.plot(omegas, As2, ls='--', marker='x', markevery=5, label='d')

    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
    #ax.set_xlim([-20, 10])
    ax.legend()
    fig.savefig('figs/Aud.png', dpi=300)

def main_TrS():
    omegas, TrS1, _ = np.loadtxt('data/lih_3A1_u_TrS.dat').T
    omegas, TrS2, _ = np.loadtxt('data/lih_3A1_d_TrS.dat').T

    fig, ax = plt.subplots()
    ax.plot(omegas, TrS1, ls='--', marker='x', markevery=20, label='u')
    ax.plot(omegas, TrS2, ls='--', marker='x', markevery=20, label='d')

    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel("Tr$\Sigma$ (eV)")
    ax.set_xlim([-20, 10])
    ax.legend()
    fig.savefig('figs/TrSud.png', dpi=300)

if __name__ == '__main__':
    # h5fname = 'lih_3A'
    # suffix = ''

    main_A()
    main_TrS()
