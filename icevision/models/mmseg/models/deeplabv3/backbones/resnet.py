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


class MMSegDeeplabBackboneConfig(MMSegBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="deeplabv3", **kwargs)


base_config_path = mmseg_configs_path / "deeplabv3"
base_weights_url = "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3"

resnet101_d16_mg124 = MMSegDeeplabBackboneConfig(
    backbone_type="R101-D16-MG124",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3/deeplabv3_r101-d16-mg124_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_40k_cityscapes/deeplabv3_r101-d16-mg124_512x1024_40k_cityscapes_20200908_005644-67b0c992.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes/deeplabv3_r101-d16-mg124_512x1024_80k_cityscapes_20200908_005644-57bb8425.pth",
        },
    ],
)

resnet101_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-101-D8",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_480x480_40k_pascal_context.py",
            "crop_size": (480, 480),
            "lr_schd": "40k",
            "pre_training_dataset": "pascal_context",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_480x480_40k_pascal_context/deeplabv3_r101-d8_480x480_40k_pascal_context_20200911_204118-1aa27336.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_480x480_40k_pascal_context_59.py",
            "crop_size": (480, 480),
            "lr_schd": "40k",
            "pre_training_dataset": "pascal_context_59",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_480x480_40k_pascal_context_59/deeplabv3_r101-d8_480x480_40k_pascal_context_59_20210416_110332-cb08ea46.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context.py",
            "crop_size": (480, 480),
            "lr_schd": "80k",
            "pre_training_dataset": "pascal_context",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context/deeplabv3_r101-d8_480x480_80k_pascal_context_20200911_170155-2a21fff3.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context_59.py",
            "crop_size": (480, 480),
            "lr_schd": "80k",
            "pre_training_dataset": "pascal_context_59",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context_59/deeplabv3_r101-d8_480x480_80k_pascal_context_59_20210416_113002-26303993.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x1024_40k_cityscapes/deeplabv3_r101-d8_512x1024_40k_cityscapes_20200605_012241-7fd3f799.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_20k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "20k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_20k_voc12aug/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_40k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "40k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_40k_voc12aug/deeplabv3_r101-d8_512x512_40k_voc12aug_20200613_161432-0017d784.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_4x4_160k_coco-stuff164k.py",
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "coco-stuff164k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_160k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_160k_coco-stuff164k_20210709_155402-f035acfd.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_4x4_20k_coco-stuff10k.py",
            "crop_size": (512, 512),
            "lr_schd": "20k",
            "pre_training_dataset": "coco-stuff10k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_20k_coco-stuff10k/deeplabv3_r101-d8_512x512_4x4_20k_coco-stuff10k_20210821_043025-c49752cb.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k.py",
            "crop_size": (512, 512),
            "lr_schd": "320k",
            "pre_training_dataset": "coco-stuff164k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_4x4_40k_coco-stuff10k.py",
            "crop_size": (512, 512),
            "lr_schd": "40k",
            "pre_training_dataset": "coco-stuff10k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_40k_coco-stuff10k/deeplabv3_r101-d8_512x512_4x4_40k_coco-stuff10k_20210821_043305-636cb433.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k.py",
            "crop_size": (512, 512),
            "lr_schd": "80k",
            "pre_training_dataset": "coco-stuff164k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k_20210709_201252-13600dc2.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "80k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k/deeplabv3_r101-d8_512x512_80k_ade20k_20200615_021256-d89c7fa4.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_769x769_40k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_769x769_40k_cityscapes/deeplabv3_r101-d8_769x769_40k_cityscapes_20200606_113809-c64f889f.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r101-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_769x769_80k_cityscapes/deeplabv3_r101-d8_769x769_80k_cityscapes_20200607_013353-60e95418.pth",
        },
    ],
)

resnet101b_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-101B-D8",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3/deeplabv3_r101b-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101b-d8_512x1024_80k_cityscapes/deeplabv3_r101b-d8_512x1024_80k_cityscapes_20201226_171821-8fd49503.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3/deeplabv3_r101b-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101b-d8_769x769_80k_cityscapes/deeplabv3_r101b-d8_769x769_80k_cityscapes_20201226_190843-9142ee57.pth",
        },
    ],
)

resnet18_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-18-D8",
    pre_trained_variants=[
        {
            "default": True,
            "config_path": "deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r18-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r18-d8_769x769_80k_cityscapes/deeplabv3_r18-d8_769x769_80k_cityscapes_20201225_021506-6452126a.pth",
        },
    ],
)

resnet18b_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-18B-D8",
    pre_trained_variants=[
        {
            "default": True,
            "config_path": "deeplabv3/deeplabv3_r18b-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r18b-d8_512x1024_80k_cityscapes/deeplabv3_r18b-d8_512x1024_80k_cityscapes_20201225_094144-46040cef.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r18b-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r18b-d8_769x769_80k_cityscapes/deeplabv3_r18b-d8_769x769_80k_cityscapes_20201225_094144-fdc985d9.pth",
        },
    ],
)

resnet50_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-50-D8",
    pre_trained_variants=[
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes/deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth",
        },
        {
            "default": True,
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes/deeplabv3_r50-d8_512x1024_80k_cityscapes_20200606_113404-b92cfdd4.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_160k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_160k_ade20k/deeplabv3_r50-d8_512x512_160k_ade20k_20200615_123227-5d0ee427.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "20k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12aug/deeplabv3_r50-d8_512x512_20k_voc12aug_20200617_010906-596905ef.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug.py",
            "crop_size": (512, 512),
            "lr_schd": "40k",
            "pre_training_dataset": "voc12aug",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug/deeplabv3_r50-d8_512x512_40k_voc12aug_20200613_161546-2ae96e7e.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_4x4_160k_coco-stuff164k.py",
            "crop_size": (512, 512),
            "lr_schd": "160k",
            "pre_training_dataset": "coco-stuff164k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_4x4_160k_coco-stuff164k/deeplabv3_r50-d8_512x512_4x4_160k_coco-stuff164k_20210709_163016-49f2812b.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_4x4_20k_coco-stuff10k.py",
            "crop_size": (512, 512),
            "lr_schd": "20k",
            "pre_training_dataset": "coco-stuff10k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_4x4_20k_coco-stuff10k/deeplabv3_r50-d8_512x512_4x4_20k_coco-stuff10k_20210821_043025-b35f789d.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_4x4_320k_coco-stuff164k.py",
            "crop_size": (512, 512),
            "lr_schd": "320k",
            "pre_training_dataset": "coco-stuff164k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_4x4_320k_coco-stuff164k/deeplabv3_r50-d8_512x512_4x4_320k_coco-stuff164k_20210709_155403-51b21115.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_4x4_40k_coco-stuff10k.py",
            "crop_size": (512, 512),
            "lr_schd": "40k",
            "pre_training_dataset": "coco-stuff10k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_4x4_40k_coco-stuff10k/deeplabv3_r50-d8_512x512_4x4_40k_coco-stuff10k_20210821_043305-dc76f3ff.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k.py",
            "crop_size": (512, 512),
            "lr_schd": "80k",
            "pre_training_dataset": "coco-stuff164k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k_20210709_163016-88675c24.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_512x512_80k_ade20k.py",
            "crop_size": (512, 512),
            "lr_schd": "80k",
            "pre_training_dataset": "ade20k",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_512x512_80k_ade20k/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_769x769_40k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "40k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_769x769_40k_cityscapes/deeplabv3_r50-d8_769x769_40k_cityscapes_20200606_113723-7eda553c.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50-d8_769x769_80k_cityscapes/deeplabv3_r50-d8_769x769_80k_cityscapes_20200606_221338-788d6228.pth",
        },
    ],
)

resnet50b_d8 = MMSegDeeplabBackboneConfig(
    backbone_type="R-50B-D8",
    pre_trained_variants=[
        {
            "default": True,
            "config_path": "deeplabv3/deeplabv3_r50b-d8_512x1024_80k_cityscapes.py",
            "crop_size": (512, 1024),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50b-d8_512x1024_80k_cityscapes/deeplabv3_r50b-d8_512x1024_80k_cityscapes_20201225_155148-ec368954.pth",
        },
        {
            "config_path": "deeplabv3/deeplabv3_r50b-d8_769x769_80k_cityscapes.py",
            "crop_size": (769, 769),
            "lr_schd": "80k",
            "pre_training_dataset": "cityscapes",
            "weights_url": "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r50b-d8_769x769_80k_cityscapes/deeplabv3_r50b-d8_769x769_80k_cityscapes_20201225_155404-87fb0cf4.pth",
        },
    ],
)
