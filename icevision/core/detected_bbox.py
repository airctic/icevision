__all__ = ["DetectedBBox"]

from .bbox import BBox


class DetectedBBox(BBox):
    """Bounding Box coupled with label and confidence score.

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

    def __init__(self, xmin, ymin, xmax, ymax, score, label):
        if score > 1.0:
            raise RuntimeWarning(f"setting detection score larger than 1.0: {score}")
        if score < 0.0:
            raise RuntimeWarning(f"setting detection score lower than 0.0: {score}")
        self.score = score
        self.label = label
        super(DetectedBBox, self).__init__(xmin, ymin, xmax, ymax)

    @classmethod
    def from_xywh(cls, x, y, w, h, score, label):
        bbox = BBox.from_xywh(x, y, w, h)
        return cls(*bbox.xyxy, score, label)

    @classmethod
    def from_xyxy(cls, xl, yu, xr, yb, score, label):
        bbox = BBox.from_xyxy(xl, yu, xr, yb)
        return cls(*bbox.xyxy, score, label)

    @classmethod
    def from_relative_xcycwh(cls, xc, yc, bw, bh, img_width, img_height, score, label):
        bbox = BBox.from_relative_xcycwh(xc, yc, bw, bh, img_width, img_height)
        return cls(*bbox.xyxy, score, label)

    @classmethod
    def from_rle(cls, rle, h, w, score, label):
        bbox = BBox.from_rle(rle, h, w)
        return cls(*bbox.xyxy, score, label)
