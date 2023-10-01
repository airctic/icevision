__all__ = ["BBox"]

from icevision.imports import *
from icevision.utils import *
from .exceptions import *


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

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"(xmin:{self.xmin}, ymin:{self.ymin}, xmax:{self.xmax}, ymax:{self.ymax})>"
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, BBox):
            return self.xyxy == other.xyxy
        return False

    def __hash__(self):
        return hash(self.xyxy)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    def to_tensor(self):
        return tensor(self.xyxy, dtype=torch.float)

    def autofix(self, img_w, img_h, record_id: Optional[Any] = None) -> bool:
        """Tries to automatically fix invalid coordinates.

        # Returns
        - False if nothing was fixed (data had no problems)
        - True if data was successfully fixed
        - Raises `InvalidDataError` if unables to automatically fix the data
        """
        # conditions where data can be fixed
        if self.xmin < 0:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox xmin from {self.xmin} to 0 (Before: {self})",
                record_id=record_id,
            )
            self.xmin = max(self.xmin, 0)

        if self.ymin < 0:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox ymin from {self.ymin} to 0 (Before: ({self}))",
                record_id=record_id,
            )
            self.ymin = max(self.ymin, 0)

        if self.xmax > img_w:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox xmax from {self.xmax} to image width {img_w} (Before: {self})",
                record_id=record_id,
            )
            self.xmax = min(self.xmax, img_w)

        if self.ymax > img_h:
            autofix_log(
                "AUTOFIX-SUCCESS",
                f"Clipping bbox ymax from {self.ymax} to image height {img_h} (Before: {self})",
                record_id=record_id,
            )
            self.ymax = min(self.ymax, img_h)

        # conditions where data cannot be fixed
        if (self.xmin >= self.xmax) or (self.ymin >= self.ymax):
            msg = []
            if self.xmin >= self.xmax:
                msg += [
                    f"\tx_min:{self.xmin} is greater than or equal to x_max:{self.xmax}"
                ]
            if self.ymin >= self.ymax:
                msg += [
                    f"\ty_min:{self.ymin} is greater than or equal to y_max:{self.ymax}"
                ]

            msg = "\n".join(msg)
            raise InvalidDataError(f"Cannot auto-fix coordinates: {self}\n{msg}")

        if self.xmin < 0 or self.ymin < 0 or self.xmax > img_w or self.ymax > img_h:
            return True

        return False

    @property
    def xyxy(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def yxyx(self):
        return (self.ymin, self.xmin, self.ymax, self.xmax)

    @property
    def xywh(self):
        return (self.xmin, self.ymin, self.width, self.height)

    def relative_xcycwh(self, img_width: int, img_height: int):
        scale = np.array([img_width, img_height, img_width, img_height])
        x, y, w, h = self.xywh / scale
        xc = x + 0.5 * w
        yc = y + 0.5 * h
        return (xc, yc, w, h)

    @classmethod
    def from_xywh(cls, x, y, w, h):
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_xyxy(cls, xl, yu, xr, yb):
        return cls(xl, yu, xr, yb)

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
