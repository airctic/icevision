import pytest, torch
from mantisshrimp import *


@pytest.fixture(scope="module")
def faster_rcnn_batch():
    dataset = test_utils.sample_dataset()
    dataloader = MantisFasterRCNN.dataloader(dataset, batch_size=2)
    xb, yb = next(iter(dataloader))
    return xb, list(yb)


@pytest.fixture(scope="module")
def mask_rcnn_batch():
    dataset = test_utils.sample_dataset()
    dataloader = MantisMaskRCNN.dataloader(dataset, batch_size=2)
    xb, yb = next(iter(dataloader))
    return xb, list(yb)


@pytest.fixture(scope="module")
def batch(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def model_class(request):
    if request.param == "faster":
        return MantisFasterRCNN
    if request.param == "mask":
        return MantisMaskRCNN


@pytest.fixture()
def assert_model_preds(request):
    expected = {
        "loss_box_reg",
        "loss_rpn_box_reg",
        "loss_objectness",
        "loss_classifier",
    }

    if request.param == "mask":
        expected.add("loss_mask")

    def _inner(model, batch):
        with torch.no_grad():
            preds = model.forward(*batch)
        assert set(preds.keys()) == expected

    return _inner


@pytest.mark.parametrize(
    "model_class, batch, assert_model_preds",
    [("faster", "faster_rcnn_batch", "faster"), ("mask", "mask_rcnn_batch", "mask")],
    indirect=True,
)
def test_rcnn_simple_backbone(model_class, simple_backbone, batch, assert_model_preds):
    model = model_class(num_classes=91, backbone=simple_backbone)
    assert_model_preds(model, batch)


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class, batch, assert_model_preds",
    [("faster", "faster_rcnn_batch", "faster"), ("mask", "mask_rcnn_batch", "mask")],
    indirect=True,
)
def test_rcnn_default_backbone(model_class, batch, assert_model_preds):
    model = model_class(num_classes=91)
    assert_model_preds(model, batch)


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class, batch, assert_model_preds",
    [("faster", "faster_rcnn_batch", "faster"), ("mask", "mask_rcnn_batch", "mask")],
    indirect=True,
)
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize(
    "backbone, fpn",
    [
        # ("mobilenet", False),
        # ("vgg11", False),
        # ("vgg13", False),
        # ("vgg16", False),
        # ("vgg19", False),
        ("resnet18", False),
        ("resnet18", True),
        # ("resnet34", False),
        # ("resnet34", True),
        # ("resnet50", False),
        # ("resnet50", True),
        # these models are too big for github runners
        # "resnet101",
        # "resnet152",
        # "resnext101_32x8d",
    ],
)
def test_mask_rcnn_backbones(
    model_class, batch, assert_model_preds, backbone, fpn, pretrained
):
    backbone = model_class.get_backbone_by_name(
        name=backbone, fpn=fpn, pretrained=pretrained
    )
    model = model_class(num_classes=91, backbone=backbone)
    assert_model_preds(model, batch)
