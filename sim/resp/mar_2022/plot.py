import numpy as np
import matplotlib.pyplot as plt


def main_chi(h5fnames, suffixes, labels=None, figname="chi", circ_label="00"):
    if labels is None:
        labels = [s[1:] for s in suffixes]

    fig, ax = plt.subplots()
    for h5fname, suffix, label in zip(h5fnames, suffixes, labels):
        omegas, chi_real, chi_imag = np.loadtxt(
            f"data/{h5fname}{suffix}_chi{circ_label}.dat"
        ).T
        ax.plot(omegas, chi_real, ls="--", label=label + ", real")
        # ax.plot(omegas, chi_imag, ls='--', label=label+', imag')
    ax.set_xlabel("$\omega$ (eV)")
    ax.set_ylabel("$\chi_{" + circ_label + "}$ (eV$^{-1}$)")
    ax.legend()
    fig.savefig(f"figs/{figname}{circ_label}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    h5fnames = ["lih", "lih", "lih"]
    suffixes = ["_exact", "_qasm", "_noisy"]

    main_chi(h5fnames, suffixes, circ_label="00a")
    main_chi(h5fnames, suffixes, circ_label="01a")
