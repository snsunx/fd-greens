"""Helper functions module."""

from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

def plot_A(h5fnames: Sequence[str], 
           suffixes: Sequence[str],
           labels: Sequence[str] = None,
           linestyles: Sequence[dict] = None,
           figname: str = 'A',
           legend: bool = True):
    """Plots the spectral function."""
    n_curves = max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves
    
    fig, ax = plt.subplots()
    for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
        omegas, As = np.loadtxt(f'data/{h5fname}{suffix}_A.dat').T
        ax.plot(omegas, As, label=label, **linestyle)
    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel('$A$ (eV$^{-1}$)')
    if legend: ax.legend()
    fig.savefig(f'figs/{figname}.png', dpi=300, bbox_inches='tight')

def plot_TrS(h5fnames: Sequence[str],
             suffixes: Sequence[str],
             labels: Sequence[str] = None,
             figname: str = 'TrS',
             linestyles: Sequence[dict] = None,
             legend: bool = True):
    """Plots the trace of the self-energy."""
    n_curves = max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves
    
    fig, ax = plt.subplots()
    for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
        omegas, real, imag= np.loadtxt(f'data/{h5fname}{suffix}_TrS.dat').T
        ax.plot(omegas, real, label=label+' (real)', **linestyle)
        ax.plot(omegas, imag, label=label+' (imag)', **linestyle)
    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel('Tr$\Sigma$ (eV)')
    if legend: ax.legend()
    fig.savefig(f'figs/{figname}.png', dpi=300, bbox_inches='tight')

def plot_chi(h5fnames: Sequence[str],
             suffixes: Sequence[str],
             labels: Sequence[str] = None,
             figname: str = 'chi',
             circ_label: str = '00',
             linestyles: Sequence[dict] = None,
             legend: bool = True):
    """Plots the charge-charge response function."""
    n_curves = max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    fig, ax = plt.subplots()
    for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
        omegas, real, imag = np.loadtxt(f'data/{h5fname}{suffix}_chi{circ_label}.dat').T
        ax.plot(omegas, real, label=label+' (real)', **linestyle)
        ax.plot(omegas, imag, label=label+' (imag)', **linestyle)
    ax.set_xlabel('$\omega$ (eV)')
    ax.set_ylabel('$\chi_{'+circ_label+'}$ (eV$^{-1}$)')
    if legend: ax.legend()
    fig.savefig(f'figs/{figname}{circ_label}.png', dpi=300, bbox_inches='tight')