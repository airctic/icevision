__all__ = [
    "ImgSize",
    "open_img",
    "get_image_size",
    "get_img_size",
    "show_img",
    "plot_grid",
]

from icevision.imports import *
import PIL, imagesize

ImgSize = namedtuple("ImgSize", "width,height")


def open_img(fn, gray=False):
    color = "L" if gray else "RGB"
    return np.array(PIL.Image.open(str(fn)).convert(color))


# TODO: Deprecated
def get_image_size(filepath: Union[str, Path]) -> Tuple[int, int]:
    """
    Returns image (width, height)
    """
    logger.warning("get_image_size is deprecated, use get_img_size instead")
    return imagesize.get(filepath)


def get_img_size(filepath: Union[str, Path]) -> ImgSize:
    """
    Returns image (width, height)
    """
    return ImgSize(*imagesize.get(filepath))


def show_img(img, ax=None, show: bool = False, **kwargs):
    img = img.squeeze().copy()
    cmap = "gray" if len(img.shape) == 2 else None

    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()

    if show:
        plt.show()

    return ax


def plot_grid(
    fs: List[callable], ncols=1, figsize=None, show=False, axs_per_iter=1, **kwargs
):
    nrows = math.ceil(len(fs) * axs_per_iter / ncols)
    figsize = figsize or (12 * ncols, 12 * nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    axs = np.asarray(axs)

    if axs_per_iter == 1:
        axs = axs.flatten()
    elif axs_per_iter > 1:
        axs = axs.reshape(-1, axs_per_iter)
    else:
        raise ValueError("axs_per_iter has to be greater than 1")

    for f, ax in zip(fs, axs):
        f(ax=ax)

    plt.tight_layout()
    if show:
        plt.show()
