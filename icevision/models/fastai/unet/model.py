__all__ = ["model"]

from icevision.imports import *
from fastai.vision.learner import model_meta, _default_meta

from fastai.vision.all import *

unet_learner
CrossEntropyLossFlat
aug_transforms


def model(backbone, num_classes, img_size, channels_in=3):
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    pretrained = True  # will come from backbone config

    meta = model_meta.get(backbone, _default_meta)
    body = fastai.create_body(backbone, channels_in, pretrained, meta["cut"])
    model = fastai.models.unet.DynamicUnet(body, num_classes, img_size)

    return model
