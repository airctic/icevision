import pytest
import torchvision

from icevision.utils.model_creation import *
from icevision import models


def test_get_module_element_form_module():
    eff_det_loaded = get_module_element_form_module(models, "torchvision", "retinanet")
    assert eff_det_loaded == models.torchvision.retinanet


def test_get_backend_libs():
    backend_libs = get_backend_libs()
    assert backend_libs == ["ross", "torchvision", "fastai", "ultralytics"]


def test_get_model_types_for_backend_lib():
    torchvision_model_types = get_model_types_for_backend_lib("torchvision")
    assert torchvision_model_types == [
        "retinanet",
        "keypoint_rcnn",
        "faster_rcnn",
        "mask_rcnn",
    ]


def test_get_backbone_names():
    backbone_names = get_backbone_names("torchvision", "retinanet")
    assert backbone_names == [
        "resnet18_fpn",
        "resnet34_fpn",
        "resnet50_fpn",
        "resnet101_fpn",
        "resnet152_fpn",
        "resnext50_32x4d_fpn",
        "resnext101_32x8d_fpn",
        "wide_resnet50_2_fpn",
        "wide_resnet101_2_fpn",
    ]


def test_load_model_components():
    model_type, backbone = load_model_components(
        "torchvision", "retinanet", "resnet18_fpn"
    )
    assert model_type == models.torchvision.retinanet
    assert backbone == models.torchvision.retinanet.backbones.resnet18_fpn


def test_load_model_components_throws_error_when_backend_lib_does_not_exist():
    with pytest.raises(AttributeError):
        model_type, backbone = load_model_components(
            "error", "retinanet", "resnet18_fpn"
        )


def test_load_model_components_throws_error_when_model_type_does_not_exist():
    with pytest.raises(AttributeError):
        model_type, backbone = load_model_components(
            "torchvision", "error", "resnet18_fpn"
        )


def test_load_model_components_throws_error_when_backbone_does_not_exist():
    with pytest.raises(AttributeError):
        model_type, backbone = load_model_components(
            "torchvision", "retinanet", "error"
        )


def test_build_model():
    model_type, model = build_model(
        "torchvision",
        "retinanet",
        "resnet18_fpn",
        num_classes=1,
        backbone_config={"pretrained": False},
        model_config=None,
    )
    assert model_type == models.torchvision.retinanet
    assert isinstance(model, torchvision.models.detection.RetinaNet)


def test_build_model_without_give_backbone_config():
    model_type, model = build_model(
        "torchvision", "retinanet", "resnet18_fpn", num_classes=1
    )
    assert model_type == models.torchvision.retinanet
    assert isinstance(model, torchvision.models.detection.RetinaNet)
