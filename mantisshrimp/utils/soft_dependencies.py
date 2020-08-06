__all__ = ["HAS_FASTAI", "HAS_LIGHTNING", "HAS_ALBUMENTATIONS"]

try:
    import fastai2.vision.all

    HAS_FASTAI = True
except ImportError as e:
    if str(e) != "No module named 'fastai2'":
        raise e

    HAS_FASTAI = False


try:
    import pytorch_lightning

    HAS_LIGHTNING = True
except ImportError as e:
    if str(e) != "No module named 'pytorch_lightning'":
        raise e

    HAS_LIGHTNING = False

try:
    import albumentations

    HAS_ALBUMENTATIONS = True
except ImportError as e:
    if str(e) != "No module named 'albumentations'":
        raise e

    HAS_ALBUMENTATIONS = False
