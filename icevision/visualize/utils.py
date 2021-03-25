__all__ = [
    "draw_label",
    "bbox_polygon",
    "draw_mask",
    "as_rgb_tuple",
    "get_default_font",
]

from icevision.imports import *
from icevision.utils import *
from matplotlib import patches
from PIL import Image, ImageFont, ImageDraw
import PIL


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


def as_rgb_tuple(x: Union[np.ndarray, tuple, list, str]) -> tuple:
    "Convert np RGB values -> tuple for PIL compatibility"
    if isinstance(x, (np.ndarray, tuple, list)):
        if not len(x) == 3:
            raise ValueError(f"Expected 3 (RGB) numbers, got {len(x)}")
        if isinstance(x, np.ndarray):
            return tuple(x.astype(np.int))
        elif isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
    elif isinstance(x, str):
        return PIL.ImageColor.getrgb(x)
    else:
        raise ValueError(f"Expected {{np.ndarray|list|tuple}}, got {type(x)}")


def get_default_font() -> str:
    import requests

    font_dir = get_root_dir() / "fonts"
    font_dir.mkdir(exist_ok=True)

    font_file = font_dir / "SpaceGrotesk-Medium.ttf"
    if not font_file.exists():
        URL = "https://raw.githubusercontent.com/airctic/storage/master/SpaceGrotesk-Medium.ttf"
        logger.info(
            "Downloading default `.ttf` font file - SpaceGrotesk-Medium.ttf from {} to {}",
            URL,
            font_file,
        )
        font_file.write_bytes(requests.get(URL).content)
    return str(font_file)
