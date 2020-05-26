__all__ = ["open_img", "show_img", "grid", "grid2"]

from ..imports import *


def open_img(fn, gray=False):
    if not os.path.exists(fn):
        raise ValueError(f"File {fn} does not exists")
    color = cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(cv2.imread(str(fn)), color)


def show_img(im, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    im = im.squeeze().copy()
    cmap = "gray" if len(im.shape) == 2 else None
    ax.imshow(im, cmap=cmap)
    ax.set_axis_off()
    return ax


def grid(f, ims, figsize=None, **kwargs):
    figsize = figsize or [7 * len(ims)] * 2
    fig, axs = plt.subplots(nrows=len(ims), figsize=figsize, **kwargs)
    for fn, ax in zip(ims, axs):
        f(fn, ax=ax)
    plt.tight_layout()


def grid2(fs, figsize=None, show=False, **kwargs):
    figsize = figsize or [7 * len(fs)] * 2
    fig, axs = plt.subplots(nrows=len(fs), figsize=figsize, **kwargs)
    for f, ax in zip(fs, axs):
        f(ax=ax)
    plt.tight_layout()
    if show:
        plt.show()
