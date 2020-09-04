__all__ = ["show_annotation"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.visualize.utils import *
from matplotlib.collections import PatchCollection


# TODO: rename im to img
def show_annotation(
    img,
    labels: List[int] = None,
    bboxes: List[BBox] = None,
    masks=None,
    class_map: ClassMap = None,
    ax: plt.axes = None,
    figsize=None,
    show=False,
) -> None:
    ax = show_img(img=img, ax=ax, figsize=figsize or (10, 10))
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
            if class_map is not None:
                label = class_map.get_id(label)
            # label position
            if bbox is not None:
                x, y = bbox.x, bbox.y
            elif mask is not None:
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
