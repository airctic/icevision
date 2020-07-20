__all__ = ["open_img", "show_img", "plot_grid"]

from mantisshrimp.imports import *


def open_img(fn, gray=False):
    if not os.path.exists(fn):
        raise ValueError(f"File {fn} does not exists")
    color = cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(cv2.imread(str(fn)), color)


def show_img(img, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    img = img.squeeze().copy()
    cmap = "gray" if len(img.shape) == 2 else None
    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    return ax


def plot_grid(fs: List[callable], ncols=1, figsize=None, show=False, **kwargs):
    figsize = figsize or (12 * len(fs) / ncols, 12)
    nrows = math.ceil(len(fs) / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

    axs = np.asarray(axs)
    for f, ax in zip(fs, axs.flatten()):
        f(ax=ax)

    plt.tight_layout()
    if show:
        plt.show()
