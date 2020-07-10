__all__ = ["model"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


# TODO: Can we map to all pretrained efficiendet models? (Not backbone in imagenet)
# checkpoint = torch.load("/home/lgvaz/Desktop/efficientdet_d0-f3276ba8.pth")
# net.load_state_dict(checkpoint)
def model(
    model_name: str, num_classes: int, img_size: int, pretrained: bool = True
) -> Tuple[nn.Module, List[List[nn.Parameter]]]:
    """ Creates the model specific by model_name

    Args:
        model_name (str): Specifies the model to create, available options are: TODO
        num_classes (int): Number of classes of your dataset (including background)
        pretrained (int): If True, use a pretrained backbone (on ImageNet)

    Returns:
          nn.Module: The requested model
    """
    config = get_efficientdet_config(model_name=model_name)
    # TODO: Verify number of classes, last model layer seems to be outputing 36
    # units no matter what
    config.num_classes = num_classes  # Should we subtract one?
    config.image_size = img_size

    net = EfficientDet(config, pretrained_backbone=pretrained)
    net.class_net = HeadNet(
        config, num_outputs=num_classes, norm_kwargs=dict(eps=0.001, momentum=0.01),
    )

    # TODO: Break down param groups for backbone
    param_groups = [
        list(net.backbone.parameters()),
        list(net.fpn.parameters()),
        [*list(net.class_net.parameters()), *list(net.box_net.parameters())],
    ]
    check_all_model_params_in_groups2(net, param_groups)

    return DetBenchTrain(net, config), param_groups
