__all__ = ["model"]

from icevision.imports import *
from icevision.utils import *
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, unwrap_bench
from effdet.efficientdet import HeadNet


def model(
    model_name: str, num_classes: int, img_size: int, pretrained: bool = True
) -> nn.Module:
    """Creates the efficientdet model specified by `model_name`.

    The model implementation is by Ross Wightman, original repo
    [here](https://github.com/rwightman/efficientdet-pytorch).

    # Arguments
        model_name: Specifies the model to create. For pretrained models, check
            [this](https://github.com/rwightman/efficientdet-pytorch#models) table.
        num_classes: Number of classes of your dataset (including background).
        img_size: Image size that will be fed to the model. Must be squared and
            divisible by 64.
        pretrained: If True, use a pretrained backbone (on COCO).

    # Returns
        A PyTorch model.
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
        config,
        num_outputs=num_classes,
        norm_kwargs=dict(eps=0.001, momentum=0.01),
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
