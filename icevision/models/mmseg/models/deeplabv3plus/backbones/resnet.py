__all__ = [
    "resnet101_d8",
    "resnet101_d16_mg124",
    "resnet101b_d8",
    "resnet18_d8",
    "resnet18b_d8",
    "resnet50_d8",
    "resnet50b_d8",
]

from icevision.imports import *
from icevision.models.mmseg.utils import *

model_name = "deeplabv3plus"


class MMSegDeeplabPlusBackboneConfig(MMSegBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name=model_name, **kwargs)


base_config_path = mmseg_configs_path / f"{model_name}"
base_weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus"

resnet101_d16_mg124 = MMSegDeeplabPlusBackboneConfig(
    backbone_type="R101-D16-MG124",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes/deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-cf9ce186.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes/deeplabv3plus_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-ee6158e0.pth",
        },
    ],
)

resnet101_d8 = MMSegDeeplabPlusBackboneConfig(
    backbone_type="R-101-D8",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_480x480_40k_pascal_context.py",
            "crop_size": (480, 480),
            "lr_schd": "40k",
            "pre_training_dataset": "pascal_context",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_480x480_40k_pascal_context/deeplabv3plus_r101-d8_480x480_40k_pascal_context_20200911_165459-d3c8a29e.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_480x480_40k_pascal_context_59.py",
            "crop_size": (480, 480),
            "lr_schd": "40k",
            "pre_training_dataset": "pascal_context_59",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_480x480_40k_pascal_context_59/deeplabv3plus_r101-d8_480x480_40k_pascal_context_59_20210416_111233-ed937f15.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_480x480_80k_pascal_context.py",
            "crop_size": (480, 480),
            "lr_schd": "80k",
            "pre_training_dataset": "pascal_context",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_480x480_80k_pascal_context/deeplabv3plus_r101-d8_480x480_80k_pascal_context_20200911_155322-145d3ee8.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_480x480_80k_pascal_context_59.py",
            "crop_size": (480, 480),
            "lr_schd": "80k",
            "pre_training_dataset": "pascal_context_59",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_480x480_80k_pascal_context_59/deeplabv3plus_r101-d8_480x480_80k_pascal_context_59_20210416_111127-7ca0331d.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes/deeplabv3plus_r101-d8_512x1024_40k_cityscapes_20200605_094614-3769eecf.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_512x512_20k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "20k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_20k_voc12aug/deeplabv3plus_r101-d8_512x512_20k_voc12aug_20200617_102345-c7ff3d56.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "40k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "80k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_ade20k/deeplabv3plus_r101-d8_512x512_80k_ade20k_20200615_014139-d5730af7.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_769x769_40k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_769x769_40k_cityscapes/deeplabv3plus_r101-d8_769x769_40k_cityscapes_20200606_114304-ff414b9e.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes/deeplabv3plus_r101-d8_769x769_80k_cityscapes_20200607_000405-a7573d20.pth",
        },
    ],
)

resnet101b_d8 = MMSegDeeplabPlusBackboneConfig(
    backbone_type="R-101B-D8",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r101b-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101b-d8_512x1024_80k_cityscapes/deeplabv3plus_r101b-d8_512x1024_80k_cityscapes_20201226_190843-9c3c93a4.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3plus/deeplabv3plus_r101b-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101b-d8_769x769_80k_cityscapes/deeplabv3plus_r101b-d8_769x769_80k_cityscapes_20201226_205041-227cdf7c.pth",
        },
    ],
)

resnet18_d8 = MMSegDeeplabPlusBackboneConfig(
    backbone_type="R-18-D8",
    pre_trained_variants=[
        {
            "default": True,
            "config_path": "deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r18-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_769x769_80k_cityscapes/deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth",
        },
    ],
)

resnet18b_d8 = MMSegDeeplabPlusBackboneConfig(
    backbone_type="R-18B-D8",
    pre_trained_variants=[
        {
            "default": True,
            "config_path": "deeplabv3plus/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes_20201226_090828-e451abd9.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r18b-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18b-d8_769x769_80k_cityscapes/deeplabv3plus_r18b-d8_769x769_80k_cityscapes_20201226_151312-2c868aff.pth",
        },
    ],
)


resnet50_d8 = MMSegDeeplabPlusBackboneConfig(
    backbone_type="R-50-D8",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "20k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_20k_voc12aug/deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "40k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "80k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_769x769_40k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_769x769_40k_cityscapes/deeplabv3plus_r50-d8_769x769_40k_cityscapes_20200606_114143-1dcb0e3c.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth",
        },
    ],
)

resnet50b_d8 = MMSegDeeplabPlusBackboneConfig(
    backbone_type="R-50B-D8",
    pre_trained_variants=[
        {
            "default": True,
            "config_path": "deeplabv3plus/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes/deeplabv3plus_r50b-d8_512x1024_80k_cityscapes_20201225_213645-a97e4e43.pth",
        },
        {
            "config_path": "deeplabv3plus/deeplabv3plus_r50b-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50b-d8_769x769_80k_cityscapes/deeplabv3plus_r50b-d8_769x769_80k_cityscapes_20201225_224655-8b596d1c.pth",
        },
    ],
)
