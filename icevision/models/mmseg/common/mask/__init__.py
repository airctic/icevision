# Soft dependencies
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.fastai:
    from icevision.models.mmseg.common.mask import fastai

# if SoftDependencies.pytorch_lightning:
# from icevision.models.mmseg.common.mask import lightning
