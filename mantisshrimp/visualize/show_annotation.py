__all__ = ["show_annotation"]

from ..imports import *
from ..utils import *
from .utils import *
from matplotlib.collections import PatchCollection


def show_annotation(
    im, labels=None, bboxes=None, masks=None, ax=None, figsize=None, show=False
):
    ax = show_img(im, ax=ax, figsize=figsize or (10, 10))
    polygons, colors = [], []
    for label, bbox, mask in itertools.zip_longest(
        *[o if o is not None else [] for o in [labels, bboxes, masks]]
    ):
        color = np.random.random(3) * 0.6 + 0.4
        colors.append(color)
        if bbox is not None:
            polygons.append(bbox_polygon(bbox))
        if mask is not None:
            draw_mask(ax, mask.data.copy(), color)
        if label is not None:
            if bbox is not None:
                x, y = bbox.x, bbox.y
            if mask is not None:
                y, x = np.unravel_index(mask.data.argmax(), mask.data.shape)
            else:
                raise ValueError("Can only display labels if bboxes or masks are given")
            draw_label(ax, x=x, y=y, name=label, color=color)

    p = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.3)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor="none", edgecolors=colors, linewidths=2)
    ax.add_collection(p)

    if show:
        plt.show()
