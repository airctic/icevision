__all__ = ["Adapter", "aug_tfms", "resize_and_pad"]

import albumentations as A
from itertools import chain
from icevision.imports import *
from icevision.core import *
from icevision.tfms.transform import *


def _resize(size, ratio_resize=A.LongestMaxSize):
    return ratio_resize(size) if isinstance(size, int) else A.Resize(*size)


def resize_and_pad(
    size: Union[int, Tuple[int, int]],
    pad: A.DualTransform = partial(
        A.PadIfNeeded, border_mode=cv2.BORDER_CONSTANT, value=[124, 116, 104]
    ),
):
    height, width = (size, size) if isinstance(size, int) else size
    return [_resize(size), pad(min_height=height, min_width=width)]


def aug_tfms(
    size: Union[int, Tuple[int, int]],
    presize: Optional[Union[int, Tuple[int, int]]] = None,
    horizontal_flip: Optional[A.HorizontalFlip] = A.HorizontalFlip(),
    shift_scale_rotate: Optional[A.ShiftScaleRotate] = A.ShiftScaleRotate(),
    rgb_shift: Optional[A.RGBShift] = A.RGBShift(),
    lightning: Optional[A.RandomBrightnessContrast] = A.RandomBrightnessContrast(),
    blur: Optional[A.Blur] = A.Blur(blur_limit=(1, 3)),
    crop_fn: Optional[A.DualTransform] = partial(A.RandomSizedBBoxSafeCrop, p=0.5),
    pad: Optional[A.DualTransform] = partial(
        A.PadIfNeeded, border_mode=cv2.BORDER_CONSTANT, value=[124, 116, 104]
    ),
) -> List[A.BasicTransform]:
    """Collection of useful augmentation transforms.

    # Arguments
        size: The final size of the image. If an `int` is given, the maximum size of
            the image is rescaled, maintaing aspect ratio. If a `tuple` is given,
            the image is rescaled to have that exact size (height, width).
        presizing: Rescale the image before applying other transfroms. If `None` this
                transform is not applied. First introduced by fastai,this technique is
                explained in their book in [this](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb)
                chapter (tip: search for "Presizing").
        horizontal_flip: Flip around the y-axis. If `None` this transform is not applied.
        shift_scale_rotate: Randomly shift, scale, and rotate. If `None` this transform
                is not applied.
        rgb_shift: Randomly shift values for each channel of RGB image. If `None` this
                transform is not applied.
        lightning: Randomly changes Brightness and Contrast. If `None` this transform
                is not applied.
        blur: Randomly blur the image. If `None` this transform is not applied.
        crop_fn: Randomly crop the image. If `None` this transform is not applied.
                Use `partial` to saturate other parameters of the class.
        pad: Pad the image to `size`, squaring the image if `size` is an `int`.
            If `None` this transform is not applied. Use `partial` to sature other
            parameters of the class.

    # Returns
        A list of albumentations transforms.
    """

    height, width = (size, size) if isinstance(size, int) else size

    tfms = []
    tfms += [_resize(presize, A.SmallestMaxSize) if presize is not None else None]
    tfms += [horizontal_flip, shift_scale_rotate, rgb_shift, lightning, blur]
    # Resize as the last transforms to reduce the number of artificial artifacts created
    if crop_fn is not None:
        crop = crop_fn(height=height, width=width)
        tfms += [A.OneOrOther(crop, _resize(size), p=crop.p)]
    else:
        tfms += [_resize(size)]
    tfms += [pad(min_height=height, min_width=width) if pad is not None else None]

    tfms = [tfm for tfm in tfms if tfm is not None]

    return tfms


class Adapter(Transform):
    """Adapter that enables the use of albumentations transforms.

    # Arguments
        tfms: `Sequence` of albumentation transforms.
    """

    def __init__(self, tfms: Sequence[A.BasicTransform]):
        self.tfms_list = tfms
        self.bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"])
        self.keypoint_params = A.KeypointParams(
            format="xy", remove_invisible=False, label_fields=["keypoints_labels"]
        )
        super().__init__(
            tfms=A.Compose(
                tfms, bbox_params=self.bbox_params, keypoint_params=self.keypoint_params
            )
        )

    def apply(
        self,
        img: np.ndarray,
        labels=None,
        bboxes: List[BBox] = None,
        masks: MaskArray = None,
        iscrowds: List[int] = None,
        keypoints: List[KeyPoints] = None,
        **kwargs
    ):
        # Substitue labels with list of idxs, so we can also filter out iscrowds in case any bboxes are removed
        # TODO: Same should be done if a masks is completely removed from the image (if bboxes is not given)
        params = {"image": img}
        params["labels"] = list(range(len(labels))) if labels is not None else []
        params["bboxes"] = [o.xyxy for o in bboxes] if bboxes is not None else []

        if keypoints is not None:
            flat_tfms_list_ = _flatten_tfms(self.tfms_list)
            if get_transform(flat_tfms_list_, "RandomSizedBBoxSafeCrop") is not None:
                raise RuntimeError(
                    "RandomSizedBBoxSafeCrop is not supported for keypoints"
                )

            k = [xy for o in keypoints for xy in o.xy]
            c = [label for o in keypoints for label in o.metadata.labels]
            v = [visible for o in keypoints for visible in o.visible]
            assert len(k) == len(c) == len(v)
            params["keypoints"] = k
            params["keypoints_labels"] = c

        if masks is not None:
            params["masks"] = list(masks.data)

        if bboxes is None:
            self.tfms.processors.pop("bboxes", None)
        if keypoints is None:
            self.tfms.processors.pop("keypoints", None)

        d = self.tfms(**params)

        img_h, img_w = _get_size_without_padding(self.tfms_list, img, d["image"])
        # We use the values in d['labels'] to get what was removed by the transform
        if labels is not None:
            keep_mask = np.zeros(len(labels), dtype=bool)
            keep_mask[d["labels"]] = True
        else:
            keep_mask = None

        out = {"img": d["image"]}
        out["height"], out["width"] = img_h, img_w

        if labels is not None:
            out["labels"] = _filter_attribute(labels, keep_mask)

        if bboxes is not None:
            # TODO: quickfix from 576
            # bb = [_clip_bboxes(xyxy, img_h, img_w) for xyxy in d["bboxes"]]
            bb = [xyxy for xyxy in d["bboxes"]]
            out["bboxes"] = [BBox.from_xyxy(*xyxy) for xyxy in bb]

        if masks is not None:
            keep_masks = _filter_attribute(d["masks"], keep_mask)
            out["masks"] = MaskArray(np.array(keep_masks))

        if keypoints is not None:
            # remove_invisible=False, therefore all points getting in are also getting out
            assert len(d["keypoints"]) == len(k)

            tfmed_kpts = _remove_outside_keypoints(d["keypoints"], img_h, img_w, v)
            # flatten list of keypoints
            flat_kpts = list(chain.from_iterable(tfmed_kpts))
            # group keypoints from same instance
            group_kpts = [
                flat_kpts[i : i + len(flat_kpts) // len(keypoints)]
                for i in range(0, len(flat_kpts), len(flat_kpts) // len(keypoints))
            ]
            assert len(group_kpts) == len(keypoints)

            kpts = [
                KeyPoints.from_xyv(group_kpt, original_kpt.metadata)
                for group_kpt, original_kpt in zip(group_kpts, keypoints)
            ]
            out["keypoints"] = _filter_attribute(kpts, keep_mask)

        if iscrowds is not None:
            out["iscrowds"] = _filter_attribute(iscrowds, keep_mask)

        return out


def _filter_attribute(v: list, keep_mask: Union[List[bool], None]):
    if keep_mask is None:
        return v
    assert len(v) == len(keep_mask)
    return [o for o, keep in zip(v, keep_mask) if keep]


def _remove_outside_keypoints(tfms_kps, h, w, v):
    """Remove keypoints that are outside image dimensions."""
    v_n = v.copy()
    tra_n = tfms_kps.copy()
    for i in range(len(tfms_kps)):
        if v[i] > 0:
            v_n[i] = _check_kps_coords(tfms_kps[i], h, w)
            if v_n[i] == 1:
                v_n[i] = v[i]
        if v_n[i] == 0:
            tra_n[i] = (0, 0)
        tra_n[i] = (tra_n[i][0], tra_n[i][1], v_n[i])
    return tra_n


def _clip_bboxes(xyxy, h, w):
    """Clip bboxes coordinates that are outside image dimensions."""
    x1, y1, x2, y2 = xyxy
    if w >= h:
        pad = (w - h) // 2
        h1 = pad
        h2 = w - pad
        return (x1, max(y1, h1), x2, min(y2, h2))
    else:
        pad = (h - w) // 2
        w1 = pad
        w2 = h - pad
        return (max(x1, w1), y1, min(x2, w2), y2)


def _get_size_without_padding(
    tfms_list, before_tfm_img, after_tfm_img
) -> Tuple[int, int]:
    height, width, _ = after_tfm_img.shape

    if get_transform(tfms_list, "Pad") is not None:
        after_pad_h, after_pad_w, _ = before_tfm_img.shape

        t = get_transform(tfms_list, "SmallestMaxSize")
        if t is not None:
            presize = t.max_size
            height, width = _func_max_size(after_pad_h, after_pad_w, presize, min)

        t = get_transform(tfms_list, "LongestMaxSize")
        if t is not None:
            size = t.max_size
            height, width = _func_max_size(after_pad_h, after_pad_w, size, max)

    return height, width


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))


def _func_max_size(height, width, max_size, func):
    scale = max_size / float(func(width, height))

    if scale != 1.0:
        height, width = tuple(py3round(dim * scale) for dim in (height, width))
    return height, width


def get_transform(tfms_list, t):
    for el in tfms_list:
        if t in str(type(el)):
            return el
    return None


def _check_kps_coords(p, h, w):
    x, y = p
    if w >= h:
        pad = (w - h) // 2
        h1 = pad
        h2 = w - pad
        return int((x <= w) and (x >= 0) and (y >= h1) and (y <= h2))
    else:
        pad = (h - w) // 2
        w1 = pad
        w2 = h - pad
        return int((x <= w2) and (x >= w1) and (y >= 0) and (y <= h))


def _is_iter(o):
    try:
        i = iter(o)
        return True
    except:
        return False


def _flatten_tfms(t):
    flat = []
    for o in t:
        if _is_iter(o):
            flat += [i for i in o]
        else:
            flat.append(o)
    return flat
