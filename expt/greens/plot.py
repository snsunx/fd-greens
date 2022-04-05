import sys

sys.path.append("../../")

from fd_greens.utils import plot_A, plot_TrS, plot_chi


def main_1p6A():
    h5fnames = ["lih_1p6A", "lih_1p6A_run2"]
    suffixes = ["_d", "_d_exp_proc"]
    labels = ["Exact", "Expt"]
    annotations_A = [
        dict(x=0.23, y=0.8, s="Exact", color="C0"),
        dict(x=0.58, y=0.5, s="Expt", color="C1"),
    ]
    annotations_TrS = [
        dict(x=0.73, y=0.7, s="Exact (real)", color="xkcd:red"),
        dict(x=0.27, y=0.59, s="Exact (imag)", color="xkcd:blue"),
        dict(x=0.36, y=0.33, s="Expt (imag)", color="xkcd:rose pink"),
        dict(x=0.73, y=0.3, s="Expt (imag)", color="xkcd:azure"),
    ]

    linestyles_A = [{}, {"ls": "--", "marker": "x", "markevery": 30}]
    linestyles_TrS = [
        {"color": "xkcd:red"},
        {"color": "xkcd:blue"},
        {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:rose pink"},
        {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:azure"},
    ]

    plot_A(
        h5fnames,
        suffixes,
        labels=labels,
        annotations=annotations_A,
        linestyles=linestyles_A,
        figname="A1p6ex",
        text="annotation",
        n_curves=1,
    )
    plot_TrS(
        h5fnames,
        suffixes,
        labels=labels,
        annotations=annotations_TrS,
        linestyles=linestyles_TrS,
        figname="TrS1p6ex",
        text="annotation",
        n_curves=2,
    )


def main_3A():
    h5fnames = ["lih_3A", "lih_3A_run2"]
    suffixes = ["_d", "_d_exp_proc"]
    labels = ["Exact", "Expt"]
    annotations_A = [
        dict(x=0.56, y=0.8, s="Exact", color="C0"),
        dict(x=0.6, y=0.45, s="Expt", color="C1"),
    ]
    annotations_TrS = [
        dict(x=0.73, y=0.7, s="Exact (real)", color="xkcd:red"),
        dict(x=0.24, y=0.55, s="Exact (imag)", color="xkcd:blue"),
        dict(x=0.58, y=0.83, s="Expt (imag)", color="xkcd:rose pink"),
        dict(x=0.62, y=0.3, s="Expt (imag)", color="xkcd:azure"),
    ]
    linestyles_A = [{}, {"ls": "--", "marker": "x", "markevery": 30}]
    linestyles_TrS = [
        {"color": "xkcd:red"},
        {"color": "xkcd:blue"},
        {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:rose pink"},
        {"ls": "--", "marker": "x", "markevery": 100, "color": "xkcd:azure"},
    ]

    plot_A(
        h5fnames,
        suffixes,
        labels=labels,
        annotations=annotations_A,
        linestyles=linestyles_A,
        figname="A3ex",
        text="annotation",
        n_curves=1,
    )
    plot_TrS(
        h5fnames,
        suffixes,
        labels=labels,
        annotations=annotations_TrS,
        linestyles=linestyles_TrS,
        figname="TrS3ex",
        text="annotation",
        n_curves=2,
    )


def main_chi():
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


def main():
    main_1p6A()
    main_3A()


if __name__ == "__main__":
    main()
