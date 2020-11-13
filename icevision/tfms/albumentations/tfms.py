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
        # Substitue labels with list of idxs, so we can also filter out iscrowds in case any bboxes is removed
        # TODO: Same should be done if a masks is completely removed from the image (if bboxes is not given)
        params = {"image": img}
        params["labels"] = list(range_of(labels)) if labels is not None else []
        params["bboxes"] = [o.xyxy for o in bboxes] if bboxes is not None else []

        if keypoints is not None:
            k = [xy for o in keypoints for xy in o.xy]
            c = [label for o in keypoints for label in o.labels]
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

        out = {"img": d["image"]}
        out["height"], out["width"], _ = out["img"].shape

        # We use the values in d['labels'] to get what was removed by the transform
        if keypoints is not None:
            tfms_kps = d["keypoints"]
            assert len(tfms_kps) == len(
                k
            )  # remove_invisible=False, therefore all points getting in are also getting out
            tfms_kps_n = filter_keypoints(tfms_kps, out["height"], out["width"], v)
            l = list(chain.from_iterable(tfms_kps_n))
            l = [
                l[i : i + len(l) // len(keypoints)]
                for i in range(0, len(l), len(l) // len(keypoints))
            ]
            assert len(l) == len(keypoints)
            cl = keypoints[0].labels
            out["keypoints"] = [KeyPoints.from_xyv(k, cl) for k in l if sum(k) > 0]
        if labels is not None:
            out["labels"] = [labels[i] for i in d["labels"]]
            if keypoints is not None:
                out["labels"] = [
                    labels[i] for i, k in zip(d["labels"], l) if sum(k) > 0
                ]
        if bboxes is not None:
            out["bboxes"] = [BBox.from_xyxy(*points) for points in d["bboxes"]]
            if keypoints is not None:
                out["bboxes"] = [
                    BBox.from_xyxy(*points)
                    for points, k in zip(d["bboxes"], l)
                    if sum(k) > 0
                ]
        if masks is not None:
            keep_masks = [d["masks"][i] for i in d["labels"]]
            if keypoints is not None:
                keep_masks = [
                    d["masks"][i] for i, k in zip(d["labels"], l) if sum(k) > 0
                ]
            out["masks"] = MaskArray(np.array(keep_masks))
        if iscrowds is not None:
            out["iscrowds"] = [iscrowds[i] for i in d["labels"]]
            if keypoints is not None:
                out["iscrowds"] = [
                    iscrowds[i] for i, k in zip(d["labels"], l) if sum(k) > 0
                ]
        return out


def filter_keypoints(tfms_kps, h, w, v):
    v_n = v.copy()
    tra_n = tfms_kps.copy()
    for i in range(len(tfms_kps)):
        if v[i] > 0:
            v_n[i] = int(
                not (
                    (tfms_kps[i][0] > w)
                    or (tfms_kps[i][1] > h)
                    or (tfms_kps[i][0] * tfms_kps[i][1] < 0)
                )
            )
        if v_n[i] == 0:
            tra_n[i] = (0, 0)
        tra_n[i] = (tra_n[i][0], tra_n[i][1], v_n[i])
    return tra_n
