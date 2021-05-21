__all__ = [
    "ImgSize",
    "open_img",
    "get_image_size",
    "get_img_size",
    "show_img",
    "plot_grid",
]

from icevision.imports import *
from PIL import ExifTags

ImgSize = namedtuple("ImgSize", "width,height")

# get exif tag
for _EXIF_ORIENTATION_TAG in ExifTags.TAGS.keys():
    if PIL.ExifTags.TAGS[_EXIF_ORIENTATION_TAG] == "Orientation":
        break

# from enum import Enum

# class PILMode(Enum):
#     blah

# FIXME
def open_img(fn, gray=False) -> PIL.Image.Image:
    "Open an image from disk `fn` as a PIL Image"
    color = "L" if gray else "RGB"
    image = PIL.Image.open(str(fn))
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert(color)
    return image


# TODO: Deprecated
def get_image_size(filepath: Union[str, Path]) -> Tuple[int, int]:
    """
    Returns image (width, height)
    """
    logger.warning("get_image_size is deprecated, use get_img_size instead")
    image_size = get_img_size(filepath=filepath)
    return image_size.width, image_size.height


def get_img_size(filepath: Union[str, Path]) -> ImgSize:
    """
    Returns image (width, height)
    """
    with PIL.Image.open(filepath) as image:
        image_size = image.size

    try:
        exif = image._getexif()
        if exif is not None and exif[_EXIF_ORIENTATION_TAG] in [6, 8]:
            image_size = image_size[::-1]
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return ImgSize(*image_size)


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
