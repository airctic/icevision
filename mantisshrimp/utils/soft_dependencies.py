__all__ = ["HAS_FASTAI", "HAS_LIGHTNING"]

try:
    import fastai2.vision.all as fastai

    HAS_FASTAI = True
except ImportError:
    HAS_FASTAI = False


try:
    import pytorch_lightning as pl

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
