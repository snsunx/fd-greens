import numpy as np
import matplotlib.pyplot as plt

def main_A():
    fig, ax = plt.subplots()
    for h5fname, suffix in zip(h5fnames, suffixes):
        omegas, As = np.loadtxt(f'data/{h5fname}{suffix}_A.dat').T
        # ax.plot(omegas, As, ls='--', marker='x', markevery=5, label=suffix[1:])
        ax.plot(omegas, As, ls='--', label=suffix[1:])

    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
    #ax.set_xlim([-20, 10])
    ax.legend()
    fig.savefig(f'figs/{Afigname}.png', dpi=300)

def main_TrS():
    fig, ax = plt.subplots()
    for h5fname, suffix in zip(h5fnames, suffixes):
        omegas, ReTrS, ImTrS = np.loadtxt(f'data/{h5fname}{suffix}_TrS.dat').T
        # ax.plot(omegas, ReTrS, ls='--', marker='x', markevery=20, label=suffix[1:] + ', real')
        # ax.plot(omegas, ImTrS, ls='-.', marker='x', markevery=20, label=suffix[1:] + ', imag')
        ax.plot(omegas, ReTrS, ls='--', label=suffix[1:] + ', real')
        ax.plot(omegas, ImTrS, ls='--', label=suffix[1:] + ', imag')


    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel("Tr$\Sigma$ (eV)")
    # ax.set_xlim([-20, 10])
    ax.legend()
    fig.savefig(f'figs/{TrSfigname}.png', dpi=300)

if __name__ == '__main__':
    h5fnames = ['lih_3A1'] * 4
    suffixes = ['_u', '_d', '_utomo', '_dtomo']
    Afigname = 'Aud'
    TrSfigname = 'TrSud'

    main_A()
    main_TrS()
