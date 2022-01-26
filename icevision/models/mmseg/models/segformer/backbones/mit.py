__all__ = ["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]

from icevision.imports import *
from icevision.models.mmseg.utils import *

model_name = "segformer"


class MMSegMITBackboneConfig(MMSegBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name=model_name, **kwargs)


base_config_path = mmseg_configs_path / f"{model_name}"
base_weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/segformer"

mit_b0 = MMSegMITBackboneConfig(
    backbone_type="MIT-B0",
    pre_trained_variants=[
        {
            "default": True,
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth",
            "config_path": "segformer/segformer_mit-b0_512x512_160k_ade20k.py",
        },
        {
            "crop_size": (1024, 1024),
            "lr_schd": "160k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth",
            "config_path": "segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py",
        },
    ],
)


mit_b1 = MMSegMITBackboneConfig(
    backbone_type="MIT-B1",
    pre_trained_variants=[
        {
            "default": True,
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth",
            "config_path": "segformer/segformer_mit-b1_512x512_160k_ade20k.py",
        },
        {
            "crop_size": (1024, 1024),
            "lr_schd": "160k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth",
            "config_path": "segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes.py",
        },
    ],
)


mit_b2 = MMSegMITBackboneConfig(
    backbone_type="MIT-B2",
    pre_trained_variants=[
        {
            "default": True,
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth",
            "config_path": "segformer/segformer_mit-b2_512x512_160k_ade20k.py",
        },
        {
            "crop_size": (1024, 1024),
            "lr_schd": "160k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth",
            "config_path": "segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes.py",
        },
    ],
)


mit_b3 = MMSegMITBackboneConfig(
    backbone_type="MIT-B3",
    pre_trained_variants=[
        {
            "default": True,
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth",
            "config_path": "segformer/segformer_mit-b3_512x512_160k_ade20k.py",
        },
        {
            "crop_size": (1024, 1024),
            "lr_schd": "160k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth",
            "config_path": "segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes.py",
        },
    ],
)


mit_b4 = MMSegMITBackboneConfig(
    backbone_type="MIT-B4",
    pre_trained_variants=[
        {
            "default": True,
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth",
            "config_path": "segformer/segformer_mit-b4_512x512_160k_ade20k.py",
        },
        {
            "crop_size": (1024, 1024),
            "lr_schd": "160k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth",
            "config_path": "segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes.py",
        },
    ],
)


mit_b5 = MMSegMITBackboneConfig(
    backbone_type="MIT-B5",
    pre_trained_variants=[
        {
            "default": True,
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth",
            "config_path": "segformer/segformer_mit-b5_512x512_160k_ade20k.py",
        },
        {
            "crop_size": (640, 640),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth",
            "config_path": "segformer/segformer_mit-b5_640x640_160k_ade20k.py",
        },
        {
            "crop_size": (1024, 1024),
            "lr_schd": "160k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth",
            "config_path": "segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py",
        },
    ],
)
