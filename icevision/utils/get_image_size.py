__all__ = ["get_image_size"]

from icevision.imports import *
import imagesize


def get_image_size(filepath: Union[str, Path]) -> Tuple[int, int]:
    return imagesize.get(filepath)
