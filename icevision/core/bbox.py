__all__ = ["BBox"]

from icevision.imports import *


# TODO: Can pnts and be tuple? Same for properties like xyxy
@dataclass
class BBox:
    """Bounding Box representation.

    Should **not** be instantiated directly, instead use `from_*` methods.
    e.g. `from_xyxy`, `from_xywh`.
    Is able to transform coordinates into different formats,
    e.g. `xyxy`, `xywh`.

    # Examples

    Create from `xywh` format, and get `xyxy` coordinates.
    ```python
    bbox = BBox.from_xywh(1, 1, 4, 4)
    xyxy = bbox.xyxy
    ```
    """

    pnts: List[int]

    def __post_init__(self):
        if self.pnts:
            xl, yu, xr, yb = self.pnts
            self.x, self.y, self.h, self.w = xl, yu, (yb - yu), (xr - xl)
            self.area = self.h * self.w

    def to_tensor(self):
        return tensor(self.xyxy, dtype=torch.float)

    @property
    def xyxy(self):
        return self.pnts

    @property
    def yxyx(self):
        xmin, ymin, xmax, ymax = self.xyxy
        return [ymin, xmin, ymax, xmax]

    @property
    def xywh(self):
        return [self.x, self.y, self.w, self.h]

    def relative_xcycwh(self, img_width: int, img_height: int):
        scale = np.array([img_width, img_height, img_width, img_height])
        x, y, w, h = self.xywh / scale
        xc = x + 0.5 * w
        yc = y + 0.5 * h
        return [xc, yc, w, h]

    @classmethod
    def from_xywh(cls, x, y, w, h):
        return cls([x, y, x + w, y + h])

    @classmethod
    def from_xyxy(cls, xl, yu, xr, yb):
        return cls([xl, yu, xr, yb])

    @classmethod
    def from_relative_xcycwh(cls, xc, yc, bw, bh, img_width, img_height):
        # subtracting 0.5 goes from center to left/upper edge, adding goes to right/bottom
        pnts = [(xc - 0.5 * bw), (yc - 0.5 * bh), (xc + 0.5 * bw), (yc + 0.5 * bh)]
        # convert from relative to absolute coordinates
        scale = np.array([img_width, img_height, img_width, img_height])
        xl, yu, xr, yb = np.around(pnts * scale).astype(int).tolist()
        return cls.from_xyxy(xl, yu, xr, yb)

    @classmethod
    def from_rle(cls, rle, h, w):
        a = np.array(rle.counts, dtype=int)
        a = a.reshape((-1, 2))  # an array of (start, length) pairs
        a[:, 0] -= 1  # `start` is 1-indexed
        y0 = a[:, 0] % h
        y1 = y0 + a[:, 1]
        if np.any(y1 > h):
            # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
            y0 = 0
            y1 = h
        else:
            y0 = np.min(y0)
            y1 = np.max(y1)
        x0 = a[:, 0] // h
        x1 = (a[:, 0] + a[:, 1]) // h
        x0 = np.min(x0)
        x1 = np.max(x1)
        if x1 > w:
            # just went out of the image dimensions
            raise ValueError(f"invalid RLE or image dimensions: x1={x1} > shape[1]={w}")
        return cls.from_xyxy(x0, y0, x1, y1)
