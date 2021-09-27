__all__ = [
    "MMSegBackboneConfig",
    "mmseg_configs_path",
    "param_groups",
    "MMSegBackboneConfig",
    "create_model_config",
]

from icevision.imports import *
from icevision.utils import *
from icevision.backbones import BackboneConfig
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
        return list(
            filter(lambda x: "default" in x and x["default"], self.pre_trained_variants)
        )[0]

    def get_pre_trained_variant(self, dataset: str, lr_schd: int, crop_size: tuple):
        return list(
            filter(
                lambda x: x["pre_training_dataset"] == dataset
                and x["lr_schd"] == lr_schd
                and x["crop_size"] == crop_size,
                self.pre_trained_variants,
            )
        )[0]


def param_groups(model):
    body = model.backbone

    layers = []
    # if isinstance(body, SSDVGG):
    #     layers += [body.features]
    #     layers += [body.extra, body.l2_norm]
    # else:
    layers += [nn.Sequential(body.conv1, body.bn1)]
    layers += [getattr(body, l) for l in body.res_layers]
    layers += [model.neck]

    # if isinstance(model, SingleStageDetector):
    #     layers += [model.bbox_head]
    # elif isinstance(model, TwoStageDetector):
    #     layers += [nn.Sequential(model.rpn_head, model.roi_head)]
    # else:
    #     raise RuntimeError(
    #         "{model} must inherit either from SingleStageDetector or TwoStageDetector class"
    #     )

    _param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, _param_groups)
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

    cfg = Config.fromfile(config_path)

    # TODO: Make this more elegant
    # HACK: Make this more elegant
    # This is required to avoid making PyTorch think we want to do parallel computing
    # https://github.com/open-mmlab/mmsegmentation/issues/809
    cfg["model"]["backbone"]["norm_cfg"] = dict(type="BN", requires_grad=True)
    cfg["model"]["decode_head"]["norm_cfg"] = dict(type="BN", requires_grad=True)
    cfg["model"]["auxiliary_head"]["norm_cfg"] = dict(type="BN", requires_grad=True)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    return cfg, weights_path
