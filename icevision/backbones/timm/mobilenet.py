__all__ = ["mobilenet"]

import timm


def our_mobilenet(*args, **kwargs):
    return timm.models.mobilenetv3(*args, **kwargs)
