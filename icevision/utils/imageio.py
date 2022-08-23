__all__ = [
    "ImgSize",
    "open_img",
    "get_image_size",
    "get_img_size",
    "get_img_size_from_data",
    "image_to_numpy",
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


def open_dicom(filename) -> PIL.Image:
    bits_stored_key = 0x00280101
    photometric_inter_key = 0x00280004

    dcm = pydicom.dcmread(filename)
    bits_stored = dcm[bits_stored_key]

    img = dcm.pixel_array

    # Check the photometric interpretation
    # MONOCHROME1: Greyscale ranges from bright to dark
    # MONOCHROME2: Greyscale ranges from dark to right
    if dcm[photometric_inter_key].value == "MONOCHROME1":
        img = 2**bits_stored - img

    # Apply a VOI LUT transformation (if the image does not
    # contain the needed parameters, the image is returned
    # unchanged)
    img = pydicom.pixel_data_handlers.apply_voi_lut(img, dcm)

    img = PIL.Image.fromarray(img)

    return img


def is_tif_extension(filepath: Union[str, Path]) -> bool:
    return str(filepath).endswith(".tif") or str(filepath).endswith(".tiff")


def swap_from_chw_to_whc(data: np.ndarray):

    if len(data.shape) == 3:
        data = np.swapaxes(data, 0, 2)
    else:
        data = np.swapaxes(data.squeeze(), 0, 1)

    return data


def open_img(
    fn, gray=False, ignore_exif: bool = False
) -> Union[PIL.Image.Image, np.ndarray]:
    """
    Open an image from disk `fn`.

    TIFF:
        returns a numpy array with dimensions (w,h,c) when gray is false and (w,h) when gray is true.
    Others:
        returns a PIL Image.
    """
    if is_tif_extension(fn):
        with rasterio.open(str(fn), "r") as img:
            raw_data = img.read()
            image = swap_from_chw_to_whc(raw_data)
            if gray:
                image = image[..., 0]
    else:
        color = "L" if gray else "RGB"

        image = PIL.Image.open(str(fn))

        if not ignore_exif:
            image = PIL.ImageOps.exif_transpose(image)
        image = image.convert(color)

    return image


def open_gray_scale_image(fn):
    """
    Opens a grayscale image, stacks the channel to represent an RGB image (1 channel duplicated 3 times to form 3 channels) and returns it as a 32 bits float array.
    """
    if ".dcm" in str(fn).lower():
        img = open_dicom(str(fn))
    elif is_tif_extension(fn):
        with rasterio.open(str(fn), "r") as image:
            raw_data = image.read()
            img = swap_from_chw_to_whc(raw_data)[..., 0]
    else:
        img = PIL.Image.open(str(fn))

    img = np.dstack([img, img, img])
    img = img.astype(np.float32)
    return img


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
    Returns image size (width, height)
    """
    if ".dcm" in str(filepath).lower():
        image = open_dicom(str(filepath))
        image_size = image.size
    else:
        if is_tif_extension(filepath):
            with rasterio.open(str(filepath), "r") as image:
                image_size = (image.width, image.height)
        else:
            with PIL.Image.open(str(filepath)) as image:
                image_size = image.size
    try:
        exif = image._getexif()
        if exif is not None and exif[_EXIF_ORIENTATION_TAG] in [6, 8]:
            image_size = image_size[::-1]
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return ImgSize(*image_size)


def get_img_size_from_data(data: Union[PIL.Image.Image, np.ndarray]) -> ImgSize:
    """
    Returns image size (width, height).
    If the data is a numpy array, it is expected to be in the format (w, h) or (w, h, c).
    """
    assert isinstance(data, (PIL.Image.Image, np.ndarray))

    if isinstance(data, PIL.Image.Image):
        width, height = data.size
    else:
        width, height, *_ = data.shape

    return ImgSize(width=width, height=height)


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


def image_to_numpy(image: Union[PIL.Image.Image, np.ndarray]) -> np.ndarray:
    """
    Returns a numpy array in the format (width, height) for a grayscale image and (width, height, channels) otherwise.
    This function should be used instead of manually converting the PIL image to a numpy array to ensure the dimensions are coherent with the expected shape format used across IceVision.
    """

    if isinstance(image, np.ndarray):
        return image

    np_img = np.array(image)

    if len(np_img.shape) == 3:
        return np_img.transpose(1, 0, 2)
    else:
        return np_img.transpose(1, 0)
