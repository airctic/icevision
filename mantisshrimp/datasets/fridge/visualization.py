__all__ = ["plot_size_histogram"]

from icevision.imports import *
from icevision.core import *
from icevision import *


def plot_size_histogram(records: List[RecordType], show=True) -> List[plt.Axes]:
    heights = []
    widths = []
    for record in records:
        heights.append(record["height"])
        widths.append(record["width"])

    fig, axs = plt.subplots(ncols=2)

    def plot_hist(ax, values, title):
        ax.hist(values, bins=30)
        ax.set_title(title)

    plot_hist(axs[0], heights, "heights")
    plot_hist(axs[1], widths, "widths")

    if show:
        plt.show()

    return axs
