import h5py
from typing import Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def plot_A(
    h5fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Sequence[str] = None,
    annotations=Optional[Sequence[dict]],
    linestyles: Optional[Sequence[dict]] = None,
    figname: str = "A",
    text: Optional[str] = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the spectral function.
    
    Args:
        h5fnames: Names of the HDF5 files from which the curves are generated.
        suffixes: Suffixes of the curves.
        labels: Legend labels of the curves.
        annotations: Annotation options to be passed into ax.text.
        linestyles: The linestyles of the curves to be passed into ax.plot.
        figname: The name of the figure to be saved.
        text: Whether to add labels by legend, annotation or none.
        n_curves: Number of curves.
    """
    assert text in [None, "legend", "annotation"]

    if n_curves is None:
        n_curves = max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves):
        omegas, As = np.loadtxt(f"data/{h5fnames[i]}{suffixes[i]}_A.dat").T
        ax.plot(omegas, As, label=labels[i], **linestyles[i])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$A$ (eV$^{-1}$)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)
    fig.savefig(f"figs/{figname}.png", dpi=300, bbox_inches="tight")


def plot_TrS(
    h5fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    figname: str = "TrS",
    linestyles: Optional[Sequence[dict]] = None,
    text: str = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the trace of the self-energy.
    
    Args:
        h5fnames: Names of the HDF5 files from which the curves are generated.
        suffixes: Suffixes of the curves.
        labels: Legend labels of the curves.
        annotations: Annotation options to be passed into ax.text.
        linestyles: The linestyles of the curves to be passed into ax.plot.
        figname: The name of the figure to be saved.
        text: Whether to add labels by legend, annotation or none.
        n_curves: Number of curves.
    """
    assert text in [None, "legend", "annotation"]

    if n_curves is None:
        n_curves = 2 * max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves // 2):
        omegas, real, imag = np.loadtxt(f"data/{h5fnames[i]}{suffixes[i]}_TrS.dat").T
        ax.plot(omegas, real, label=labels[i] + " (real)", **linestyles[2 * i])
        ax.plot(omegas, imag, label=labels[i] + " (imag)", **linestyles[2 * i + 1])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("Tr$\Sigma$ (eV)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)
    fig.savefig(f"figs/{figname}.png", dpi=300, bbox_inches="tight")


def plot_chi(
    h5fnames: Sequence[str],
    suffixes: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    annotations: Optional[Sequence[str]] = None,
    figname: str = "chi",
    circ_label: str = "00",
    linestyles: Sequence[dict] = None,
    text: Optional[str] = None,
    n_curves: Optional[int] = None,
) -> None:
    """Plots the charge-charge response function.

    Args:
        h5fnames: Names of the HDF5 files from which the curves are generated.
        suffixes: Suffixes of the curves.
        labels: Legend labels of the curves.
        annotations: Annotation options to be passed into ax.text.
        linestyles: The linestyles of the curves to be passed into ax.plot.
        figname: The name of the figure to be saved.
        circ_label: The circuit label.
        text: Whether to add labels by legend, annotation or none.
        n_curves: Number of curves.
    """
    if n_curves is None:
        n_curves = 2 * max(len(h5fnames), len(suffixes))
    if labels is None:
        labels = [s[1:] for s in suffixes]
    if linestyles is None:
        linestyles = [None] * n_curves

    plt.clf()
    fig, ax = plt.subplots()
    # for h5fname, suffix, label, linestyle in zip(h5fnames, suffixes, labels, linestyles):
    for i in range(n_curves // 2):
        omegas, real, imag = np.loadtxt(
            f"data/{h5fnames[i]}{suffixes[i]}_chi{circ_label}.dat"
        ).T
        ax.plot(omegas, real, label=labels[i] + " (real)", **linestyles[2 * i])
        ax.plot(omegas, imag, label=labels[i] + " (imag)", **linestyles[2 * i + 1])
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$\chi_{" + circ_label + "}$ (eV$^{-1}$)")
    if text == "legend":
        ax.legend()
    elif text == "annotation":
        # for kwargs in annotations:
        for i in range(n_curves):
            ax.text(**annotations[i], transform=ax.transAxes)
    fig.savefig(f"figs/{figname}{circ_label}.png", dpi=300, bbox_inches="tight")


def plot_counts(
    h5fname: str, counts_name: str, circ_label: str, tomo_label: str, width: float = 0.5
) -> None:
    """Plots the QASM and experimental bitstring counts.
    
    Args:
        h5fname: The HDF5 file name.
        counts_name: Name of the bitstring counts.
        circ_label: The circuit label.
        tomo_label: The tomography label.
        width: The width of the bars in the bar chart.
    """
    h5file = h5py.File(h5fname + ".h5", "r")
    # print(circ_label, tomo_label)
    dset = h5file[f"circ{circ_label}/{tomo_label}"]
    counts = dset.attrs[counts_name]
    counts_norm = counts / np.sum(counts)
    counts_exp = dset.attrs[counts_name + "_exp_proc"]
    counts_exp_norm = counts_exp / np.sum(counts_exp)
    tvd = np.sum(np.abs(counts_norm - counts_exp_norm)) / 2

    n_qubits = int(np.log2(len(counts)))
    x = np.arange(2 ** n_qubits)
    tick_labels = ["".join(x) for x in product("01", repeat=n_qubits)]

    plt.clf()
    if n_qubits == 3:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 3, counts_norm, width / 1.5, label="QASM")
    ax.bar(x + width / 3, counts_exp_norm, width / 1.5, label="Expt")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Ratio")
    ax.set_title(f"Total Variational Distance: {tvd:.4f}")
    ax.legend()
    fig.savefig(
        f"figs/counts_{h5fname}_{circ_label}{tomo_label}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
