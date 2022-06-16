import sys

sys.path.append("../../")

from fd_greens.utils import plot_chi


def main():
    h5fnames = ["lih", "lih_run2"]
    suffixes = ["_exact", "_noisy_exp_proc"]
    labels = ["Exact", "Expt"]
    annotations = [
        dict(x=0.22, y=0.16, s="Exact (real)", color="xkcd:red"),
        dict(x=0.45, y=0.52, s="Exact (imag)", color="xkcd:blue"),
        dict(x=0.45, y=0.41, s="Expt (real)", color="xkcd:rose pink"),
        dict(x=0.12, y=0.45, s="Expt (imag)", color="xkcd:azure"),
    ]
    linestyles = [
        {"color": "xkcd:red"},
        {"color": "xkcd:blue"},
        {"ls": "--", "marker": "x", "markevery": 30, "color": "xkcd:rose pink"},
        {"ls": "--", "marker": "x", "markevery": 30, "color": "xkcd:azure"},
    ]

    plot_chi(
        h5fnames,
        suffixes,
        labels=labels,
        annotations=annotations,
        circ_label="00",
        linestyles=linestyles,
        text="annotation",
        n_curves=2,
        figname="chiex",
    )
    """
    plot_chi(
        h5fnames,
        suffixes,
        labels=labels,
        circ_label="01",
        linestyles=linestyles,
        text="annotation",
    )
    """


"""
def main_chi():
    h5fnames = ["lih", "lih_run2"]
    suffixes = ["_exact", "_noisy_exp_proc"]
    linestyles = [{}, {"ls": "--", "marker": "x", "markevery": 30}]

    plot_chi(h5fnames, suffixes, circ_label="00", linestyles=linestyles)
    plot_chi(h5fnames, suffixes, circ_label="01", linestyles=linestyles)
"""


if __name__ == "__main__":
    main()
