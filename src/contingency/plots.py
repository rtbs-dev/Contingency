import numpy as np
from typing import Any, Callable
from functools import wraps
from beartype import beartype
from .types import Params, Returns, ScoreOptions, IsolineMetric

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from cycler import cycler
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    plt = None  # type: ignore[assignment]
    Axes = Any  # type: ignore[misc,assignment]


def _check_plot() -> None:
    if not _HAS_PLOT:
        raise ImportError(
            "Optional dependencies required. "
            "(Install using `pip install contingency-tools[plot]`)"
        )


def _requires_plot(func: Callable[Params, Returns]) -> Callable[Params, Returns]:
    """decorator to enforce plotting dependencies"""

    @wraps(func)
    def wrapper(*args: Params.args, **kwargs: Params.kwargs) -> Returns:
        _check_plot()
        return func(*args, **kwargs)

    return wrapper


def _plot_isoline_func(thres, func, name, ax=None):
    if ax is None:
        ax = plt.gca()
    lines = [(t, *func(t)) for t in thres]
    for t, x, y in lines:
        (l,) = ax.plot(x, y, color="0.95", ls=("--" if t < 0 else "-"))
        cum_arclen = np.insert(
            np.cumsum(np.linalg.norm(np.diff(np.column_stack([x, y]), axis=0), axis=1)),
            0,
            0,
        )
        mid_x = np.interp(cum_arclen[-1] / 2, cum_arclen, x)
        mid_y = np.interp(cum_arclen[-1] / 2, cum_arclen, y)
        ax.annotate(
            f"{t:0.1f}",
            # xy=((fp.max()+fp.min())/2, (tp.max()+tp.min())/2),
            xy=(mid_x - 0.02, mid_y - 0.02),
            color="0.8",
            bbox=dict(facecolor="white", linewidth=0, alpha=0.5),
        )
    ax.annotate(
        name,
        xy=(1.02, lines[0][-1][-1] - 0.01),
        color="0.9",
        annotation_clip=False,
    )


@beartype
@_requires_plot
def PR_contour(
    r: float | None = None,
    isolines: IsolineMetric | None = "F",
    show_lower_bound: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Generate a nice-looking contour plot for Precision vs. Recall
    
    For an example, see the [Tutorial](getting-started/02-tutorial/#optional-plotting-utilities)
    
    REQUIRES optional `contingency[plot]` dependencies! See [Installation](getting-started/01-installation).

    Args:
        r: class imbalance ratio (required for mcc and show_lower_bound, e.g. from `Contingency.imbalance`)
        isolines: which score metric isolines to show (if any)
        show_lower_bound: For imbalanced data, shade below the Precision floor, given some Recall. 
        ax: existing matplotlib.pyplot.Axes to use (if any)

    Returns:
        Axes

    """

    if ax is None:
        ax = plt.gca()

    def _fm_isoline(t):
        recall_fm = np.linspace(t**2, 1)
        precis_fm = np.minimum(1, t**2 / recall_fm)
        return recall_fm, precis_fm

    def _f1_isoline(t):
        recall_f1 = np.linspace(t / (2 - t), 1)
        precis_f1 = np.minimum(t * recall_f1 / (2 * recall_f1 - t), 1)
        return recall_f1, precis_f1

    def _mcc_isoline(t):
        p_bound = lambda rec: (
            (
                1
                + (r * t**2 + t * np.sqrt((r * t) ** 2 + 4 * r * rec * (1 - rec)))
                / (2 * rec)
            )
            / (r + 1)
        )
        if t < 0:
            r_mcc = np.linspace(1e-6, 1)
            p_mcc = p_bound(r_mcc)
        elif np.isclose(t, 0):
            r_mcc = np.linspace(0, 1)
            p_mcc = np.ones_like(r_mcc) / (r + 1)
        else:
            r_mcc = np.linspace(t**2 * (r + 1) / (t**2 + r), 1)
            p_mcc = p_bound(r_mcc)
        return r_mcc, p_mcc

    match isolines:
        case "G":
            thres = np.linspace(0.2, 0.8, num=4)
            _plot_isoline_func(thres, _fm_isoline, r"F-M", ax=ax)
        case "F":
            thres = np.linspace(0.2, 0.8, num=4)
            _plot_isoline_func(thres, _f1_isoline, r"$F_1$", ax=ax)
        case "mcc":
            if r is None:
                raise ValueError(
                    "MCC isolines are undefined if class imbalance `r` is None"
                )
            thres = np.array([-1, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
            _plot_isoline_func(thres, _mcc_isoline, r"MCC", ax=ax)
        case None:
            pass

    if show_lower_bound:
        if r is None:
            raise ValueError(
                "Lower bound on precision is undefined for class imbalance `r=None`"
            )
        ax.fill_between(
            (grid := np.linspace(1e-4, 1)),
            grid / (grid + r),
            facecolor="none",
            edgecolor="0.9",
            hatch="//",
        )

    ax.set(
        ylim=(0, 1.01),
        xlim=(0, 1.01),
        ylabel="Precision",
        xlabel="Recall",
        aspect=1,
    )
    return ax


@beartype
@_requires_plot
def ROC_contour(
    r:float,
    isolines: IsolineMetric | None = "F",
    ax: Axes | None = None,
):
    """Generate a nice-looking contour plot for ROC (TPR vs FNR)

    Args:
        r: class imbalance ratio (required, e.g. from `Contingency.imbalance`)
        isolines: which score metric isolines to show (if any)
        ax: existing matplotlib.pyplot.Axes to use (if any)

    """

    if ax is None:
        ax = plt.gca()

    def _fm_isoline(t):
        fpr_fm = np.linspace(0, np.minimum(1, (1 - t**2) / (t**2 * r)))
        tpr_fm = (t**2 + np.sqrt(t**4 + 4 * t**2 * r * fpr_fm)) / 2.0
        return fpr_fm, tpr_fm

    def _f1_isoline(t):
        fpr_f1 = np.linspace(0, np.minimum(1, 2 * (1 - t) / (t * r)))
        tpr_f1 = np.minimum(t * (r * fpr_f1 + 1) / (2 - t), 1)
        return fpr_f1, tpr_f1

    def _mcc_isoline(t):
        A = t**2 + r
        B = lambda f: 2 * r * f * (t**2 - 1) - t**2 * (r + 1)
        C = lambda f: r * f * (f * (r * t**2 + 1) - t**2 * (r + 1))

        tpr_bound = lambda f: (
            (-B(f) + np.sign(t) * np.sqrt(B(f) ** 2 - 4 * A * C(f))) / (2 * A)
        )

        if t < 0:
            fpr = np.linspace(np.maximum(0, t**2 * (r + 1) / (r * (t**2 + 1))), 1)
            tpr = tpr_bound(fpr)
        elif np.isclose(t, 0):
            fpr = np.linspace(0, 1)
            tpr = np.linspace(0, 1)
        else:
            fpr = np.linspace(0, np.minimum(1, (r - t**2) / (r * (t**2 + 1))))
            tpr = tpr_bound(fpr)
        return fpr, tpr

    match isolines:
        case "G":
            thres = np.linspace(0.2, 0.8, num=4)
            _plot_isoline_func(thres, _fm_isoline, r"F-M", ax=ax)
        case "F":
            thres = np.linspace(0.2, 0.8, num=4)
            _plot_isoline_func(thres, _f1_isoline, r"$F_1$", ax=ax)
        case "mcc":
            thres = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
            _plot_isoline_func(thres, _mcc_isoline, r"MCC", ax=ax)
        case None:
            pass

    ax.set(
        ylim=(0, 1.01),
        xlim=(-0.01, 1.0),
        ylabel="True Positive Rate",
        xlabel="False Positive Rate",
        aspect=1,
    )
    return ax


@beartype
class ContingencyPlots:
    """Plotting accessor for [`Contingent`][contingency.contingent.Contingent] objects"""

    def __init__(self, data: "contingency.Contingent") -> None:
        _check_plot()
        self.M = data

    def pr(self, *args, **kwargs):
        """alias for [`precision_recall`][contingency.plots.ContingencyPlots.precision_recall]"""
        return self.precision_recall(*args, **kwargs)

    def roc(self, *args, **kwargs):
        """alias for [`receiver_operating_characteristic`][contingency.plots.ContingencyPlots.receiver_operating_characteristic]"""
        return self.receiver_operating_characteristic(*args, **kwargs)

    def precision_recall(
        self,
        isolines: IsolineMetric | None = "F",
        show_lower_bound:bool=False,
        optimize: ScoreOptions | list[ScoreOptions] | None = None,
        ax:Axes|None=None, **step_kws
    ) -> Axes:
        """Create Precision-recall curve with optional isolines and optimums.

        Args:
          isolines: which score metric isolines to show (if any)
          show_lower_bound: For imbalanced data, shade below the Precision floor, given some Recall. 
            ax: existing matplotlib.pyplot.Axes to use (if any)
          optimize: optional score or list of scores to plot their optimal location 
          ax: existing matplotlib.pyplot.Axes to use (if any)
          **step_kws: options to pass to matplotlib.pyplot.step

        Returns:

        """
        ax = PR_contour(
            self.M.imbalance,
            isolines=isolines,
            show_lower_bound=show_lower_bound,
            ax=ax,
        )
        ax.step(self.M.recall, self.M.precision, where="post", **step_kws)

        if optimize is None:
            optimize = []
        elif isinstance(optimize, str):
            optimize = [optimize]

        marker_cycler = iter(
            cycler(marker=['D', 's','o', '^']) *
            cycler(fillstyle=['full', 'none'])
        )
        color = ax.get_lines()[-1].get_color()
        
        for metric in optimize:
            met_opt = self.M.optimize(metric)
            ax.plot(
                met_opt.recall,
                met_opt.precision,
                linestyle='none',
                color=color,
                label=metric + f"* thres={met_opt.weights[0]:.2f}",
                **next(marker_cycler)  #type:ignore
            )

        return ax

    def receiver_operating_characteristic(
        self,
        isolines: IsolineMetric | None = "mcc",
        optimize: ScoreOptions | list[ScoreOptions] | None = None,
        ax:Axes|None=None, **plot_kws
    )-> Axes:
        """Plot the Receiver Operating Characteristic with optional isolines and optimums.

        Args:
          isolines: which score metric isolines to show (if any)
          optimize: optional score or list of scores to plot their optimal location 
          ax: existing matplotlib.pyplot.Axes to use (if any)
          **plot_kws: options to pass to matplotlib.pyplot.plot
                    
         """
        ax = ROC_contour(self.M.imbalance, isolines=isolines, ax=ax)
        ax.plot(self.M.FPR, self.M.TPR, **plot_kws)

        if optimize is None:
            optimize = []
        elif isinstance(optimize, str):
            optimize = [optimize]
        marker_cycler = iter(
            cycler(marker=['o', 'D', 's', '^']) *
            cycler(fillstyle=['full', 'none'])
        )
        color = ax.get_lines()[-1].get_color()
        for metric in optimize:
            met_opt = self.M.optimize(metric)
            ax.plot(
                met_opt.FPR,
                met_opt.TPR,
                color=color,
                label=metric + f"* thres={met_opt.weights[0]:.2f}",
                linestyle='none',
                **next(marker_cycler) #type:ignore
            )
        return ax
