# The draw functions are based from:
# https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/visualization.py
# https://github.com/fizyr/keras-maskrcnn/blob/master/keras_maskrcnn/utils/visualization.py

__all__ = [
    "draw_sample",
    "draw_record",
    "draw_pred",
    "draw_bbox",
    "draw_mask",
    "draw_keypoints",
]

from icevision.imports import *
from icevision.data import *
from icevision.core import *


def draw_sample(
    sample,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    display_keypoints: bool = True,
):
    img = sample["img"].copy()
    if denormalize_fn is not None:
        img = denormalize_fn(img)

    for label, bbox, mask, keypoints in itertools.zip_longest(
        sample.get("labels", []),
        sample.get("bboxes", []),
        sample.get("masks", []),
        sample.get("keypoints", []),
    ):
        color = (np.random.random(3) * 0.6 + 0.4) * 255
        color = tuple(color.astype(int).tolist())

        if display_mask and mask is not None:
            img = draw_mask(img=img, mask=mask, color=color)
        if display_bbox and bbox is not None:
            img = draw_bbox(img=img, bbox=bbox, color=color)
        if display_label and label is not None:
            img = draw_label(
                img=img,
                label=label,
                bbox=bbox,
                mask=mask,
                class_map=class_map,
                color=color,
            )
        if display_keypoints and keypoints is not None:
            img = draw_keypoints(img=img, kps=keypoints, color=color)

    return img


def draw_label(
    img: np.ndarray,
    label: int,
    color,
    class_map: Optional[ClassMap] = None,
    bbox=None,
    mask=None,
):
    # finds label position based on bbox or mask
    if bbox is not None:
        x, y, _, _ = bbox.xyxy
    elif mask is not None:
        y, x = np.unravel_index(mask.data.argmax(), mask.data.shape)
    else:
        x, y = 0, 0

    if class_map is not None:
        caption = class_map.get_id(label)
    else:
        caption = str(label)

    return _draw_label(img=img, caption=caption, x=int(x), y=int(y), color=color)


def _draw_label(
    img: np.ndarray,
    caption: str,
    x: int,
    y: int,
    color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
):
    """Draws a caption above the box in an image."""
    y -= 10
    w, h = cv2.getTextSize(caption, font, fontScale=font_scale, thickness=1)[0]

    # make the coords of the box with a small padding of two pixels
    # check if the box_pt2 is inside the image otherwise invert the label box (meaning the label box will be inside the bounding box)
    if (y - h - 2) > 0:
        box_pt1, box_pt2 = (x, y + 10), (x + w + 2, y - h - 2)
    else:
        box_pt1, box_pt2 = (x, y + h + 22), (x + w + 2, y + 10)

    cv2.rectangle(img, box_pt1, box_pt2, color, cv2.FILLED)

    label_pt = (box_pt1[0], box_pt1[1] - 10)
    cv2.putText(img, caption, label_pt, font, font_scale, (240, 240, 240), 2)

    return img


def draw_record(
    record,
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    display_keypoints: bool = True,
):
    sample = record.load()
    return draw_sample(
        sample=sample,
        class_map=class_map,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
        display_keypoints=display_keypoints,
    )


def draw_pred(
    img: np.ndarray,
    pred: dict,
    class_map: Optional[ClassMap] = None,
    denormalize_fn: Optional[callable] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
):
    sample = pred.copy()
    sample["img"] = img
    return draw_sample(
        sample=sample,
        class_map=class_map,
        denormalize_fn=denormalize_fn,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
    )


def draw_bbox(
    img: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int],
    gap: bool = True,
):
    """Draws a box on an image with a given color.
    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
    """

    # Calculate image dimensions
    dims = sorted(img.shape, reverse=True)

    # corner thickness is linearly correlated with the smaller image dimension.
    # We use the smaller image dimension rather than image area so as to avoid
    # overly thick lines for large non-square images prior to transforming
    # images. We set lower and upper bounds for corner thickness.
    min_corner = 1
    max_corner = 15
    corner_thickness = int(0.005 * dims[1] + min_corner)
    if corner_thickness > max_corner:
        corner_thickness = int(max_corner)

    corner_length = int(0.021 * dims[1] + 2.25)

    # inner thickness of bboxes with corners
    inner_thickness = int(1 + 0.0005 * dims[1])

    # bbox thickness of bboxes without corners
    min_bbox = 1
    max_bbox = 8
    bbox_thickness = int(0.0041 * dims[1] - 0.0058)
    if bbox_thickness < min_bbox:
        bbox_thickness = min_bbox
    if bbox_thickness > max_bbox:
        bbox_thickness = int(max_bbox)

    if gap == False:
        xyxy = tuple(np.array(bbox.xyxy, dtype=int))
        cv2.rectangle(img, xyxy[:2], xyxy[2:], color, bbox_thickness, cv2.LINE_AA)
        return img

    xmin, ymin, xmax, ymax = tuple(np.array(bbox.xyxy, dtype=int))

    points = [0] * 12
    points[0] = (xmin, ymin + corner_length)
    points[1] = (xmin, ymin)
    points[2] = (xmin + corner_length, ymin)

    points[3] = (xmax - corner_length, ymin)
    points[4] = (xmax, ymin)
    points[5] = (xmax, ymin + corner_length)

    points[6] = (xmax, ymax - corner_length)
    points[7] = (xmax, ymax)
    points[8] = (xmax - corner_length, ymax)

    points[9] = (xmin + corner_length, ymax)
    points[10] = (xmin, ymax)
    points[11] = (xmin, ymax - corner_length)

    if (
        ymax - (ymin + 4 * corner_length) < corner_length
        or xmax - (xmin + 4 * corner_length) < corner_length
    ):
        for i in range(4):
            cv2.line(
                img,
                points[i * 3 + 1],
                points[10 - 3 * i],
                color,
                bbox_thickness,
                cv2.LINE_4,
            )
        for i in range(2):
            cv2.line(
                img,
                points[6 * i + 1],
                points[i * 6 + 4],
                color,
                bbox_thickness,
                cv2.LINE_4,
            )
    else:
        for i in range(2):
            for j in range(2):
                cv2.line(
                    img,
                    points[i * 6 + j * 4],
                    points[i * 6 + j * 4 + 1],
                    color,
                    corner_thickness,
                    cv2.LINE_AA,
                )
                cv2.line(
                    img,
                    points[i * 3 + 1],
                    points[10 - 3 * i],
                    color,
                    inner_thickness,
                    cv2.LINE_4,
                )
        for i in range(2):
            for j in range(2):
                cv2.line(
                    img,
                    points[i * 6 + j * 2 + 1],
                    points[i * 6 + j * 2 + 2],
                    color,
                    corner_thickness,
                    cv2.LINE_4,
                )
                cv2.line(
                    img,
                    points[6 * i + 1],
                    points[i * 6 + 4],
                    color,
                    inner_thickness,
                    cv2.LINE_AA,
                )

    return img


def draw_mask(
    img: np.ndarray, mask: MaskArray, color: Tuple[int, int, int], blend: float = 0.5
):
    color = np.asarray(color, dtype=int)
    # draw mask
    mask_idxs = np.where(mask.data)
    img[mask_idxs] = blend * img[mask_idxs] + (1 - blend) * color

    # draw border
    border = mask.data - cv2.erode(mask.data, np.ones((7, 7), np.uint8), iterations=1)
    border_idxs = np.where(border)
    img[border_idxs] = color

    return img


def draw_keypoints(
    img: np.ndarray,
    kps: KeyPoints,
    color: Tuple[int, int, int],
):
    x, y, v = kps.x, kps.y, kps.visible

    # calculate scaling for points and connections
    img_h, img_w, _ = img.shape
    img_area = img_h * img_w
    dynamic_size = int(0.01867599 * (img_area ** 0.4422045))
    dynamic_size = max(dynamic_size, 1)

    # draw connections
    if kps.metadata is not None and kps.metadata.connections is not None:
        for connection in kps.metadata.connections:
            if v[connection.p1] > 0 and v[connection.p2] > 0:
                cv2.line(
                    img,
                    (int(x[connection.p1]), int(y[connection.p1])),
                    (int(x[connection.p2]), int(y[connection.p2])),
                    color=connection.color,
                    thickness=dynamic_size,
                )

    # draw points
    for x_c, y_c in zip(x[v > 0], y[v > 0]):
        cv2.circle(
            img,
            (int(round(x_c)), int(round(y_c))),
            radius=dynamic_size,
            color=color,
            thickness=-1,
        )

    return img
