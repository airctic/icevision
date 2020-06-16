import pytest, torch
from mantisshrimp import *


@pytest.fixture(scope="session")
def batch():
    dataset = test_utils.sample_dataset()
    dataloader = MantisMaskRCNN.dataloader(dataset, batch_size=2)
    xb, yb = next(iter(dataloader))
    return xb, list(yb)


# Same backbones as in faster rcnn


@pytest.mark.slow
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize(
    "backbone, fpn",
    [
        (None, True),
        ("mobilenet", False),
        ("vgg11", False),
        ("vgg13", False),
        ("vgg16", False),
        ("vgg19", False),
        ("resnet18", False),
        ("resnet34", False),
        ("resnet50", False),
        ("resnet18", True),
        ("resnet34", True),
        ("resnet50", True),
        # these models are too big for github runners
        # "resnet101",
        # "resnet152",
        # "resnext101_32x8d",
    ],
)
def test_faster_rcnn_nonfpn_backbones(batch, backbone, fpn, pretrained):
    if backbone is not None:
        backbone = MantisMaskRCNN.get_backbone_by_name(
            name=backbone, fpn=fpn, pretrained=pretrained
        )
    model = MantisMaskRCNN(n_class=91, backbone=backbone)
    with torch.no_grad():
        preds = model.forward(*batch)

    print(set(preds.keys()))
    assert set(preds.keys()) == set(
        ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    )
