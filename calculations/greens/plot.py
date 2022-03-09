import numpy as np
import matplotlib.pyplot as plt

def main_A(h5fnames, suffixes, labels=None, figname='A'):
    if labels is None:
        labels = [s[1:] for s in suffixes]
    fig, ax = plt.subplots()
    for h5fname, suffix, label in zip(h5fnames, suffixes, labels):
        omegas, As = np.loadtxt(f'data/{h5fname}{suffix}_A.dat').T
        ax.plot(omegas, As, ls='--', label=label)
    ax.set_xlabel('$\omega$ (eV)')
    # ax.set_ylabel("Absorption spectra (eV$^{-1}$)")
    ax.set_ylabel("A (eV$^{-1}$)")
    ax.legend()
    fig.savefig(f'figs/{figname}.png', dpi=300, bbox_inches='tight')

def main_TrS(h5fname, suffixes, labels=None, figname='TrS'):
    if labels is None:
        labels = [s[1:] for s in suffixes]

    fig, ax = plt.subplots()
    for h5fname, suffix, label in zip(h5fnames, suffixes, labels):
        omegas, ReTrS, ImTrS = np.loadtxt(f'data/{h5fname}{suffix}_TrS.dat').T
        ax.plot(omegas, ReTrS, ls='--', label=label+', real')
        ax.plot(omegas, ImTrS, ls='--', label=label+', imag')


    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel("Tr$\Sigma$ (eV)")
    # ax.set_xlim([-20, 10])
    ax.legend()
    fig.savefig(f'figs/{figname}.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    h5fnames = ['lih_1d6A', 'lih_1d6A']
    suffixes = ['_d', '_dnoisy']
    labels = ['Exact', 'Noisy']

    main_A(h5fnames, suffixes, labels=labels, figname='A1d6')
    main_TrS(h5fnames, suffixes, labels=labels, figname='TrS1d6')
