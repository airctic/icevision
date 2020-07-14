__all__ = ["model"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, unwrap_bench
from effdet.efficientdet import HeadNet


def model(
    model_name: str, num_classes: int, img_size: int, pretrained: bool = True
) -> nn.Module:
    """ Creates the model specific by model_name

    Args:
        model_name (str): Specifies the model to create, available options are: TODO
        num_classes (int): Number of classes of your dataset (including background)
        pretrained (int): If True, use a pretrained backbone (on COCO)

    Returns:
          nn.Module: The requested model
    """
    config = get_efficientdet_config(model_name=model_name)

    net = EfficientDet(config, pretrained_backbone=False)
    if pretrained:
        weights_url = _weights_url[model_name]
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, map_location=torch.device("cpu")
        )
        net.load_state_dict(state_dict)

    config.num_classes = num_classes
    config.image_size = img_size
    net.class_net = HeadNet(
        config, num_outputs=num_classes, norm_kwargs=dict(eps=0.001, momentum=0.01),
    )

    # TODO: Break down param groups for backbone
    def param_groups_fn(model: nn.Module) -> List[List[nn.Parameter]]:
        unwrapped = unwrap_bench(model)

        layers = [
            unwrapped.backbone,
            unwrapped.fpn,
            nn.Sequential(unwrapped.class_net, unwrapped.box_net),
        ]
        param_groups = [list(layer.parameters()) for layer in layers]
        check_all_model_params_in_groups2(model, param_groups)

        return param_groups

    model_bench = DetBenchTrain(net, config)
    model_bench.param_groups = MethodType(param_groups_fn, model_bench)

    return model_bench


_weights_url = {
    "tf_efficientdet_lite0": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite0-f5f303a9.pth",
    "tf_efficientdet_d0": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0-d92fd44f.pth",
    "efficientdet_d0": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-f3276ba8.pth",
    "tf_efficientdet_d1": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1-4c7ebaf2.pth",
    "efficientdet_d1": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth",
    "tf_efficientdet_d2": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2-cb4ce77d.pth",
    "tf_efficientdet_d3": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3-b0ea2cbc.pth",
    "tf_efficientdet_d4": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4-5b370b7a.pth",
    "tf_efficientdet_d5": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5-ef44aea8.pth",
    "tf_efficientdet_d6": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6-51cb0132.pth",
    "tf_efficientdet_d7": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth",
}
