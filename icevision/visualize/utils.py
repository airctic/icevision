__all__ = ["draw_label", "bbox_polygon", "draw_mask"]

from icevision.imports import *
from matplotlib import patches


def draw_label(ax, x, y, name, color, fontsize=18):
    ax.text(
        x + 1,
        y - 2,
        name,
        fontsize=fontsize,
        color="white",
        va="bottom",
        bbox=dict(facecolor=color, edgecolor=color, pad=2, alpha=0.9),
    )


def bbox_polygon(bbox):
    bx, by, bw, bh = bbox.xywh
    poly = np.array([[bx, by], [bx, by + bh], [bx + bw, by + bh], [bx + bw, by]])
    return patches.Polygon(poly)


def draw_mask(ax, mask, color):
    color_mask = np.ones((*mask.shape, 3)) * color
    ax.imshow(np.dstack((color_mask, mask * 0.5)))
    ax.contour(mask, colors=[color_mask[0, 0, :]], alpha=0.4)
