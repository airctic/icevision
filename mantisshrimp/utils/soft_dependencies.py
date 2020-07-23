__all__ = ["HAS_FASTAI", "HAS_LIGHTNING"]

try:
    import fastai2.vision.all as fastai

    HAS_FASTAI = True
except ImportError as e:
    if str(e) != "No module named 'fastai2'":
        raise e

    HAS_FASTAI = False


try:
    import pytorch_lightning as pl

    HAS_LIGHTNING = True
except ImportError as e:
    if str(e) != "No module named 'pytorch_lightning'":
        raise e

    HAS_LIGHTNING = False
