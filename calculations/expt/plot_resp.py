import sys
sys.path.append('../../src')
from helpers import plot_chi

import numpy as np
import matplotlib.pyplot as plt
"""
def main_chi(h5fnames, suffixes, labels=None, figname='chi', circ_label='00', linestyles=None):
    if labels is None:
        labels = [s[1:] for s in suffixes]

    fig, ax = plt.subplots()
    for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
        omegas, chi_real, chi_imag = np.loadtxt(f'data/{h5fname}{suffix}_chi{circ_label}.dat').T
        ax.plot(omegas, chi_real, label=label+', real', **linestyle)
        ax.plot(omegas, chi_imag, label=label+', imag', **linestyle)
    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel('$\chi_{'+circ_label+'}$ (eV$^{-1}$)')
    # ax.legend()
    fig.savefig(f'figs/{figname}{circ_label}.png', dpi=300, bbox_inches='tight')
"""
def main_chi():
    h5fnames = ['lih', 'lih_run2']
    suffixes = ['_exact', '_noisy_exp_proc']
    linestyles = [{}, {'ls': '--', 'marker': 'x', 'markevery': 30}]
    
    plot_chi(h5fnames, suffixes, circ_label='00', linestyles=linestyles)
    plot_chi(h5fnames, suffixes, circ_label='01', linestyles=linestyles)

if __name__ == '__main__':
    # h5fnames = ['lih', 'lih_run2']
    # suffixes = ['_exact', '_noisy_exp_proc']
    # linestyles = [{}, {'ls': '--', 'marker': 'x', 'markevery': 30}]

    # main_chi(h5fnames, suffixes, circ_label='00', linestyles=linestyles)
    # main_chi(h5fnames, suffixes, circ_label='01', linestyles=linestyles)
    main_chi()
