import cv2
import numpy as np
import random

from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.crops import functional as F
from albumentations.augmentations.bbox_utils import union_of_bboxes
from albumentations.augmentations.geometric import functional as FGeometric


class CustomRandomSizedBBoxSafeCrop(DualTransform):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        max_ar_distortion: maximum difference between crop area aspect ratio and mosaic tile aspect ratio, value 0 means
         that the boxes have exactly the same aspect ratio, suggested range (0.1, 0.5)
        num_tries: maximum number of tries in crop area selection to meet the aspect ratio distortion criteria
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height,
        width,
        erosion_rate=0.0,
        interpolation=cv2.INTER_LINEAR,
        max_ar_distortion=0.2,
        num_tries=10,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.max_ar_distortion = max_ar_distortion
        self.num_tries = num_tries
        self.interpolation = interpolation
        self.erosion_rate = erosion_rate

    def apply(
        self,
        img,
        crop_height=0,
        crop_width=0,
        h_start=0,
        w_start=0,
        interpolation=cv2.INTER_LINEAR,
        **params
    ):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return FGeometric.resize(crop, self.height, self.width, interpolation)

    def apply_to_bbox(
        self,
        bbox,
        crop_height=0,
        crop_width=0,
        h_start=0,
        w_start=0,
        rows=0,
        cols=0,
        **params
    ):
        return F.bbox_random_crop(
            bbox, crop_height, crop_width, h_start, w_start, rows, cols
        )

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if (
            len(params["bboxes"]) == 0
        ):  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = (
                img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            )
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        bboxes = params["bboxes"]
        # n_boxes = np.random.randint(len(bboxes))
        # bboxes = [bboxes[n_boxes]]
        # get union of all bboxes
        x, y, x2, y2 = union_of_bboxes(
            width=img_w,
            height=img_h,
            bboxes=bboxes,
            erosion_rate=self.erosion_rate,
        )
        # find bigger region until aspect ratio distortion criteria is met or max tries reached
        target_aspect_ratio = self.width / self.height
        for _ in range(self.num_tries):
            bx, by = x * random.random(), y * random.random()
            bx2, by2 = (
                x2 + (1 - x2) * random.random(),
                y2 + (1 - y2) * random.random(),
            )
            bw, bh = bx2 - bx, by2 - by
            crop_height = img_h if bh >= 1.0 else int(img_h * bh)
            crop_width = img_w if bw >= 1.0 else int(img_w * bw)
            crop_aspect_ratio = crop_width / crop_height
            if abs(crop_aspect_ratio - target_aspect_ratio) < self.max_ar_distortion:
                break
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {
            "h_start": h_start,
            "w_start": w_start,
            "crop_height": crop_height,
            "crop_width": crop_width,
        }

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "erosion_rate", "interpolation")
