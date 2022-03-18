from typing import Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np

def plot_A(h5fnames: Sequence[str], 
           suffixes: Sequence[str],
           labels: Sequence[str] = None,
           annotations = Optional[Sequence[dict]],
           linestyles: Optional[Sequence[dict]] = None,
           figname: str = 'A',
           text: Optional[str] = None,
           n_curves: Optional[int] = None):
    """Plots the spectral function."""
    assert text in [None, 'legend', 'annotation']
    if n_curves is None:
        n_curves = max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves
    
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves):
        omegas, As = np.loadtxt(f'data/{h5fnames[i]}{suffixes[i]}_A.dat').T
        ax.plot(omegas, As, label=labels[i], **linestyles[i])
    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel('$A$ (eV$^{-1}$)')
    if text == 'legend': 
        ax.legend()
    elif text == 'annotation':
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)
    fig.savefig(f'figs/{figname}.png', dpi=300, bbox_inches='tight')

def plot_TrS(h5fnames: Sequence[str],
             suffixes: Sequence[str],
             labels: Optional[Sequence[str]] = None,
             annotations: Optional[Sequence[str]] = None,
             figname: str = 'TrS',
             linestyles: Optional[Sequence[dict]] = None,
             text: str = None,
             n_curves: Optional[int] = None):
    """Plots the trace of the self-energy."""
    assert text in [None, 'legend', 'annotation']

    if n_curves is None:
        n_curves = 2 * max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves
    
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves//2):
        omegas, real, imag= np.loadtxt(f'data/{h5fnames[i]}{suffixes[i]}_TrS.dat').T
        ax.plot(omegas, real, label=labels[i]+' (real)', **linestyles[2*i])
        ax.plot(omegas, imag, label=labels[i]+' (imag)', **linestyles[2*i+1])
    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel('Tr$\Sigma$ (eV)')
    if text == 'legend': 
        ax.legend()
    elif text == 'annotation':
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)
    fig.savefig(f'figs/{figname}.png', dpi=300, bbox_inches='tight')

def plot_chi(h5fnames: Sequence[str],
             suffixes: Sequence[str],
             labels: Optional[Sequence[str]] = None,
             annotations: Optional[Sequence[str]] = None,
             figname: str = 'chi',
             circ_label: str = '00',
             linestyles: Sequence[dict] = None,
             text: Optional[str] = None,
             n_curves: Optional[int] = None):
    """Plots the charge-charge response function."""
    if n_curves is None:
        n_curves = 2 * max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves//2):
        omegas, real, imag = np.loadtxt(f'data/{h5fnames[i]}{suffixes[i]}_chi{circ_label}.dat').T
        ax.plot(omegas, real, label=labels[i]+' (real)', **linestyles[2*i])
        ax.plot(omegas, imag, label=labels[i]+' (imag)', **linestyles[2*i+1])
    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel('$\chi_{'+circ_label+'}$ (eV$^{-1}$)')
    if text == 'legend': 
        ax.legend()
    elif text == 'annotation':
        # for kwargs in annotations:
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)
    fig.savefig(f'figs/{figname}{circ_label}.png', dpi=300, bbox_inches='tight')