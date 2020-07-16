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
        if not config.url:
            raise RuntimeError(f"No pretrained weights for {model_name}")
        state_dict = torch.hub.load_state_dict_from_url(
            config.url, map_location=torch.device("cpu")
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
