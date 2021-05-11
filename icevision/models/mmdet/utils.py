__all__ = [
    "MMDetBackboneConfig",
    "mmdet_configs_path",
    "param_groups",
    "MMDetBackboneConfig",
    "create_model_config",
]

from icevision.imports import *
from icevision.utils import *
from icevision.backbones import BackboneConfig
from icevision.models.mmdet.download_configs import download_mmdet_configs
from mmdet.models.detectors import *
from mmcv import Config
from mmdet.models.backbones.ssd_vgg import SSDVGG


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
    if isinstance(body, SSDVGG):
        layers += [body.features]
        layers += [body.extra, body.l2_norm]
    else:
        layers += [nn.Sequential(body.conv1, body.bn1)]
        layers += [getattr(body, l) for l in body.res_layers]
        layers += [model.neck]

    if isinstance(model, SingleStageDetector):
        layers += [model.bbox_head]
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
):

    model_name = backbone.model_name
    config_path = backbone.config_path
    weights_url = backbone.weights_url

    # download weights
    if pretrained and weights_url:
        save_dir = Path(checkpoints_path) / model_name
        save_dir.mkdir(exist_ok=True, parents=True)

        fname = Path(weights_url).name
        weights_path = save_dir / fname

        if not weights_path.exists() or force_download:
            download_url(url=weights_url, save_path=str(weights_path))

    cfg = Config.fromfile(config_path)

    return cfg, weights_path
