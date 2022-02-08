__all__ = [
    "MMDetBackboneConfig",
    "mmdet_configs_path",
    "param_groups",
    "MMDetBackboneConfig",
    "create_model_config",
]

from numpy import False_
from icevision.imports import *
from icevision.utils import *
from icevision.backbones import BackboneConfig
from icevision.models.mmdet.download_configs import download_mmdet_configs
from mmdet.models.detectors import *
from mmcv import Config
from mmdet.models.backbones.ssd_vgg import SSDVGG
from mmdet.models.backbones.csp_darknet import CSPDarknet
from mmdet.models.backbones.swin import SwinTransformer
from mmdet.models.backbones.hourglass import HourglassNet


mmdet_configs_path = download_mmdet_configs()


class MMDetBackboneConfig(BackboneConfig):
    def __init__(self, model_name, config_path, weights_url):
        self.model_name = model_name
        self.config_path = config_path
        self.weights_url = weights_url
        self.pretrained: bool

    def __call__(self, pretrained: bool = True) -> "MMDetBackboneConfig":
        self.pretrained = pretrained
        return self


def param_groups(model):
    body = model.backbone

    layers = []

    # add the backbone
    if isinstance(body, SSDVGG):
        layers += [body.features]
    elif isinstance(body, CSPDarknet):
        layers += [body.stem.conv.conv, body.stem.conv.bn]
        layers += [body.stage1, body.stage2, body.stage3, body.stage4]

    elif isinstance(body, HourglassNet):
        layers += [
            body.stem,
            body.hourglass_modules,
            body.inters,
            body.conv1x1s,
            body.out_convs,
            body.remap_convs,
            body.relu,
        ]

    elif isinstance(body, SwinTransformer):
        layers += [
            body.patch_embed.adap_padding,
            body.patch_embed.projection,
            body.patch_embed.norm,
            body.drop_after_pos,
            body.stages,
        ]

        # Swin backbone for two-stage detector has norm0 attribute
        if hasattr(body, "norm0"):
            layers += [body.norm0]

        layers += [body.norm1, body.norm2, body.norm3]
    else:
        layers += [nn.Sequential(body.conv1, body.bn1)]
        layers += [getattr(body, l) for l in body.res_layers]

    # add the neck module if it exists (DETR doesn't have a neck module)
    if hasattr(model, "neck"):
        layers += [model.neck]

    # add the head
    if isinstance(model, SingleStageDetector):
        layers += [model.bbox_head]

        # YOLACT has mask_head and segm_head
        if hasattr(model, "mask_head"):
            layers += [model.mask_head]
        if hasattr(model, "segm_head"):
            layers += [model.segm_head]

    elif isinstance(model, TwoStageDetector):
        layers += [nn.Sequential(model.rpn_head, model.roi_head)]
    else:
        raise RuntimeError(
            "{model} must inherit either from SingleStageDetector or TwoStageDetector class"
        )

    _param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, _param_groups)
    return _param_groups


def create_model_config(
    backbone: MMDetBackboneConfig,
    pretrained: bool = True,
    checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
    force_download=False,
    cfg_options=None,
):

    model_name = backbone.model_name
    config_path = backbone.config_path
    weights_url = backbone.weights_url

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

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    return cfg, weights_path
