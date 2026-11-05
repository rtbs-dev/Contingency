import numpy as np
from typing import Literal

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes
except ImportError:
    _has_plot = False
else:
    _has_plot = True


def PR_contour(ax: matplotlib.axes.Axes | None = None):
    """Generate a nice-looking contour plot for Precision vs. Recall

    For an example, see the [Tutorial](getting-started/02-tutorial/#optional-plotting-utilities)

    REQUIRES optional `contingency[plot]` dependencies! See [Installation](getting-started/01-installation).
    """
    if not _has_plot:
        raise ImportError("Optional contingiency[plot] dependencies required.")

    if ax is None:
        ax = plt.gca()
    thres = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for t in thres:
        recall_f1 = np.linspace(t / (2 - t), 1.0)
        recall_fm = np.linspace(t**2, 1.0)
        prec_f1 = t * recall_f1 / (2 * recall_f1 - t)
        prec_fm = t**2 / recall_fm

        (l,) = ax.plot(recall_f1, prec_f1, color="0.8")
        (l,) = ax.plot(recall_fm, prec_fm, color="0.95")
        ax.annotate(
            f"{t:0.1f}",
            xy=(t - 0.02, t - 0.02),
            color="0.8",
            bbox=dict(facecolor="white", linewidth=0, alpha=0.5),
        )

    ax.annotate(r"$F_1$", xy=(1.01, 0.2 / (2 - 0.2) - 0.01), color="0.8")
    ax.annotate(r"F-M", xy=(1.01, 0.2**2 - 0.01), color="0.9")
    ax.set(ylim=(0, 1.1), xlim=(0, 1.1), ylabel="Precision", xlabel="Recall")


def ROC_contour(
    r,
    isolines: list[Literal["f1", "fm"]] | None = ["f1"],
    ax: matplotlib.axes.Axes | None = None,
):
    """Generate a nice-looking contour plot for ROC (TPR vs FNR)

    Args:
        r: class imbalance ratio (required, e.g. from `Contingency.imbalance`)
        isolines: which score metric isolines to show (if any)
        ax: existing matplotlib.pyplot.Axes to use (if any)

    """
    if ax is None:
        ax = plt.gca()
    thres = np.linspace(0.2, 0.8, num=4)

    def _fm_isoline_fpr(t):
        fpr_fm = np.linspace(0, np.minimum(1, (1 - t**2) / (t**2 * r)))
        tpr_fm = (t**2 + np.sqrt(t**4 + 4 * t**2 * r * fpr_fm)) / 2.0
        return fpr_fm, tpr_fm

    def _f1_isoline_fpr(t):
        fpr_f1 = np.linspace(0, np.minimum(1, 2 * (1 - t) / (t * r)))
        tpr_f1 = np.minimum(t * (r * fpr_f1 + 1) / (2 - t), 1)
        return fpr_f1, tpr_f1

    def _plot_isoline_func(func, name):
        lines = [(t, *func(t)) for t in thres]
        for t, fp, tp in lines:
            (l,) = ax.plot(fp, tp, color="0.95")
            ax.annotate(
                f"{t:0.1f}",
                # xy=((fp.max()+fp.min())/2, (tp.max()+tp.min())/2),
                xy=(np.median(fp), np.median(tp)),
                color="0.8",
                bbox=dict(facecolor="white", linewidth=0, alpha=0.5),
            )
        ax.annotate(name, xy=(1.01, lines[0][-1][-1] - 0.01), color="0.9")

    if isolines is None:
        isolines = []

    if "fm" in isolines:
        _plot_isoline_func(_fm_isoline_fpr, r"F-M")
    if "f1" in isolines:
        _plot_isoline_func(_f1_isoline_fpr, r"$F_1$")

    ax.set(
        ylim=(0, 1.1),
        xlim=(-0.01, 1.1),
        ylabel="True Positive Rate",
        xlabel="False Positive Rate",
        aspect=1,
    )
    return ax
