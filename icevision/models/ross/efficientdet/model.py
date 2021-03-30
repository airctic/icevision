__all__ = ["model"]

from icevision.imports import *
from icevision.utils import *
from icevision.models.ross.efficientdet.utils import *
from icevision.models.ross.efficientdet.backbones import *
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, unwrap_bench
from effdet import create_model_from_config
from effdet.efficientdet import HeadNet


def model(
    backbone: EfficientDetBackboneConfig,
    num_classes: int,
    img_size: int,
) -> nn.Module:
    """Creates the efficientdet model specified by `model_name`.

    The model implementation is by Ross Wightman, original repo
    [here](https://github.com/rwightman/efficientdet-pytorch).

    # Arguments
        backbone: Specifies the backbone to use create the model. For pretrained models, check
            [this](https://github.com/rwightman/efficientdet-pytorch#models) table.
        num_classes: Number of classes of your dataset (including background).
        img_size: Image size that will be fed to the model. Must be squared and
            divisible by 128.

    # Returns
        A PyTorch model.
    """
    model_name = backbone.model_name
    config = get_efficientdet_config(model_name=model_name)
    config.image_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    model_bench = create_model_from_config(
        config,
        bench_task="train",
        bench_labeler=True,
        num_classes=num_classes - 1,
        pretrained=backbone.pretrained,
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

    model_bench.param_groups = MethodType(param_groups_fn, model_bench)

    return model_bench
