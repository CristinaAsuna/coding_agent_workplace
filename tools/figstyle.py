"""Shared figure conventions for discovery-lab topics (matplotlib, static PNG).

Categorical colors are a fixed-order palette validated for color-vision
deficiency on a light surface: give each condition a slot and keep it for the
whole topic. Text stays in ink colors; series colors are for marks only.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "lab_sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("lab_diverging", ["#2a78d6", "#f0efec", "#e34948"])
MAX_BYTES = 1024 * 1024


def setup():
    """Apply the shared rcParams; call once before plotting."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_MUTED,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False,
    })


def seed_lines(ax, x, ys, label=None, color=None, naive=None, naive_label="naive expectation"):
    """Plot one thin line per seed and a bold mean; ys has shape (seeds, len(x)).

    naive: optional curve the prior belief predicts, drawn as a gray dashed line.
    """
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    (mean_line,) = ax.plot(x, ys.mean(axis=0), lw=2.0, color=color, label=label, zorder=3)
    for y in ys:
        ax.plot(x, y, lw=0.8, alpha=0.35, color=mean_line.get_color(), zorder=2)
    if naive is not None:
        ax.plot(x, naive, ls="--", lw=1.2, color=INK_MUTED, label=naive_label, zorder=1)
    return mean_line


def representative_seed(values, seeds=None):
    """Return the seed whose primary metric is closest to the median across seeds."""
    values = np.asarray(values, dtype=float)
    index = int(np.argmin(np.abs(values - np.median(values))))
    return seeds[index] if seeds is not None else index


def save_fig(fig, path, max_bytes=MAX_BYTES):
    """Save a figure as PNG under the size limit and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{path} is {size / 1024:.0f} KB > {max_bytes / 1024:.0f} KB; lower dpi or simplify")
    return path
