__all__ = [
    "MMSegBackboneConfig",
    "mmseg_configs_path",
    "param_groups",
    "param_groups_default",
    "MMSegBackboneConfig",
    "create_model_config",
]

from icevision.imports import *
from icevision.utils import *
from icevision.models.backbone_config import BackboneConfig
from icevision.core.exceptions import PreTrainedVariantNotFound
from icevision.models.mmseg.download_configs import download_mmseg_configs
from mmseg.models.segmentors import *
from mmcv import Config

# from mmdet.models.backbones.ssd_vgg import SSDVGG


mmseg_configs_path = download_mmseg_configs()


class MMSegBackboneConfig(BackboneConfig):
    def __init__(self, model_name: str, backbone_type: str, pre_trained_variants: dict):
        self.model_name = model_name
        self.backbone_type = backbone_type
        self.pre_trained_variants = pre_trained_variants
        self.pretrained: bool

    def __call__(self, pretrained: bool = True) -> "MMSegBackboneConfig":
        self.pretrained = pretrained
        return self

    def get_default_pre_trained_variant(self):
        """Fetch the default pre-trained variant for the backbone configuration

        Returns:
            dict: pre-trained variant information
        """

        default_variant = list(
            filter(lambda x: "default" in x and x["default"], self.pre_trained_variants)
        )

        if not len(default_variant):
            raise PreTrainedVariantNotFound

        return default_variant[0]

    def get_pre_trained_variant(self, dataset: str, lr_schd: int, crop_size: tuple):
        """Fetch a specific pre-trained variant for the backbone configuration

        Args:
            dataset (str): dataset the variant was trained on
            lr_schd (int): learning schedule
            crop_size (tuple): input crop size

        Returns:
            dict: pre-trained variant information
        """

        variant = list(
            filter(
                lambda x: x["pre_training_dataset"] == dataset
                and x["lr_schd"] == lr_schd
                and x["crop_size"] == crop_size,
                self.pre_trained_variants,
            )
        )

        if not len(variant):
            raise PreTrainedVariantNotFound

        return variant[0]


def param_groups(model):

    layers = [*model.children()]

    _param_groups = [list(group.parameters()) for group in layers]

    check_all_model_params_in_groups2(model, _param_groups)
    return _param_groups


def param_groups_default(model):

    _param_groups = [
        p for p in model.parameters() if p.requires_grad
    ]  # equivalent to fastai.trainable_params

    # check_all_model_params_in_groups2(model, [_param_groups])
    return _param_groups


def create_model_config(
    model_name: str,
    config_path: str,
    weights_url: str,
    pretrained: bool = True,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
    cfg_options=None,
):

    # download weights
    weights_path = None
    if pretrained and weights_url:
        save_dir = Path(checkpoints_path) / model_name
        save_dir.mkdir(exist_ok=True, parents=True)

        fname = Path(weights_url).name
        weights_path = save_dir / fname

        if not weights_path.exists() or force_download:
            download_url(url=weights_url, save_path=str(weights_path))

    cfg = Config.fromfile(mmseg_configs_path / config_path)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    return cfg, weights_path
