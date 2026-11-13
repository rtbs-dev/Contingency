import numpy as np
from typing import Literal

try:
    import matplotlib.pyplot as plt
    import matplotlib.axes
except ImportError:
    _has_plot = False
else:
    _has_plot = True

def _plot_isoline_func(thres, func, name, ax=None):
    if ax is None:
        ax = plt.gca()
    lines = [(t, *func(t)) for t in thres]
    for t, x, y in lines:
        (l,) = ax.plot(x, y, color="0.95")
        cum_arclen = np.insert(np.cumsum(np.linalg.norm(np.diff(np.column_stack([x,y]), axis=0), axis=1)),0,0)
        mid_x = np.interp(cum_arclen[-1]/2, cum_arclen, x)
        mid_y = np.interp(cum_arclen[-1]/2, cum_arclen, y)
        ax.annotate(
            f"{t:0.1f}",
            # xy=((fp.max()+fp.min())/2, (tp.max()+tp.min())/2),
            xy=(mid_x-0.02, mid_y-0.02),
            color="0.8",
            bbox=dict(facecolor="white", linewidth=0, alpha=0.5),
        )
    ax.annotate(
        name,
        xy=(1.02, lines[0][-1][-1] - 0.01),
        color="0.9",
        annotation_clip=False,
    )



def PR_contour(    
    r:float|None=None, 
    isolines: list[Literal["f1", "fm", 'mcc']] | None = ["f1"],
    ax: matplotlib.axes.Axes | None = None
):
    """Generate a nice-looking contour plot for Precision vs. Recall

    For an example, see the [Tutorial](getting-started/02-tutorial/#optional-plotting-utilities)

    REQUIRES optional `contingency[plot]` dependencies! See [Installation](getting-started/01-installation).
    """
    if not _has_plot:
        raise ImportError("Optional contingiency[plot] dependencies required.")

    if ax is None:
        ax = plt.gca()
    
    def _fm_isoline(t):
        recall_fm = np.linspace(t**2,1)
        precis_fm = np.minimum(1,t**2/recall_fm)
        return recall_fm, precis_fm

    def _f1_isoline(t):
        recall_f1 = np.linspace(t / (2 - t),1)
        precis_f1 = np.minimum(t * recall_f1 / (2 * recall_f1 - t), 1)
        return recall_f1, precis_f1

    def _mcc_isoline(t):
        p_bound = lambda rec: (1+(r*t**2+t*np.sqrt((r*t)**2+4*r*rec*(1-rec)))/(2*rec))/(r+1)
        if t<0: 
            r_mcc = np.linspace(1e-6, 1) 
            p_mcc = p_bound(r_mcc) 
        elif np.isclose(t, 0): 
            r_mcc = np.linspace(0,1)
            p_mcc = np.ones_like(r_mcc)/(r+1)
        else:
            r_mcc = np.linspace(t**2*(r+1)/(t**2+r), 1) 
            p_mcc = p_bound(r_mcc)
        return r_mcc, p_mcc

    if isolines is None:
        isolines = []

    if "fm" in isolines:
        thres = np.linspace(0.2, 0.8, num=4)
        _plot_isoline_func(thres, _fm_isoline, r"F-M", ax=ax)
    if "f1" in isolines:
        thres = np.linspace(0.2, 0.8, num=4)
        _plot_isoline_func(thres, _f1_isoline, r"$F_1$", ax=ax)
    if "mcc" in isolines:
        thres = np.array([-1., -0.2,0., 0.2, 0.4, 0.6, 0.8])
        if r is None: 
            raise ValueError("MCC isolines are undefined if class imbalance `r` is None")
        _plot_isoline_func(thres, _mcc_isoline, r"MCC", ax=ax)

    ax.set(
        ylim=(0, 1.01),
        xlim=(0, 1.01),
        ylabel="Precision",
        xlabel="Recall",
        aspect=1,
    )
    return ax


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

    def _fm_isoline(t):
        fpr_fm = np.linspace(0, np.minimum(1, (1 - t**2) / (t**2 * r)))
        tpr_fm = (t**2 + np.sqrt(t**4 + 4 * t**2 * r * fpr_fm)) / 2.0
        return fpr_fm, tpr_fm

    def _f1_isoline(t):
        fpr_f1 = np.linspace(0, np.minimum(1, 2 * (1 - t) / (t * r)))
        tpr_f1 = np.minimum(t * (r * fpr_f1 + 1) / (2 - t), 1)
        return fpr_f1, tpr_f1

    def _mcc_isoline(t):
        A = t**2+r
        B = lambda f: 2*r*f*(t**2-1)-t**2*(r+1)
        C = lambda f: r*f*(f*(r*t**2+1)-t**2*(r+1))

        tpr_bound = lambda f: (-B(f)+np.sign(t)*np.sqrt(B(f)**2-4*A*C(f)))/(2*A)

        if t<0: 
            fpr = np.linspace(np.maximum(0,t**2*(r+1)/(r*(t**2+1))), 1)
            tpr = tpr_bound(fpr)
        elif np.isclose(t, 0): 
            fpr = np.linspace(0,1)
            tpr = np.linspace(0,1)
        else:
            fpr = np.linspace(0,np.minimum(1, (r-t**2)/(r*(t**2+1))))
            tpr = tpr_bound(fpr)
        return fpr,tpr

    if isolines is None:
        isolines = []

    if "fm" in isolines:
        thres = np.linspace(0.2, 0.8, num=4)
        _plot_isoline_func(thres, _fm_isoline, r"F-M", ax=ax)
    if "f1" in isolines:
        thres = np.linspace(0.2, 0.8, num=4)
        _plot_isoline_func(thres, _f1_isoline, r"$F_1$", ax=ax)
    if "mcc" in isolines:
        thres = np.array([-0.8, -0.6, -0.4, -0.2,0., 0.2, 0.4, 0.6, 0.8])
        _plot_isoline_func(thres, _mcc_isoline, r"MCC", ax=ax)

    ax.set(
        ylim=(0, 1.01),
        xlim=(-0.01, 1.),
        ylabel="True Positive Rate",
        xlabel="False Positive Rate",
        aspect=1,
    )
    return ax
