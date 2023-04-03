import pytest
import torch
import icevision as iv


def test_save_icevision_checkpoint_with_all_parameters(
    faster_rcnn_model_mmdet_with_num_classes_1, tmp_path
):
    checkpoint_save_path = tmp_path / "checkpoints"
    checkpoint_save_path.mkdir()

    # setup parameters
    optimizer = torch.optim.Adam(
        faster_rcnn_model_mmdet_with_num_classes_1.parameters()
    )
    classes = ["test"]
    model_name = "mmdet.faster_rcnn"
    img_size = 512
    backbone_name = "resnet50_fpn_1x"

    iv.models.checkpoint.save_icevision_checkpoint(
        model=faster_rcnn_model_mmdet_with_num_classes_1,
        filename=str(checkpoint_save_path / "test_checkpoint"),
        optimizer=optimizer,
        model_name=model_name,
        backbone_name=backbone_name,
        classes=classes,
        img_size=img_size,
    )


def test_save_icevision_checkpoint_throws_type_error_if_meta_is_not_dict(
    faster_rcnn_model_mmdet_with_num_classes_1, tmp_path
):
    checkpoint_save_path = tmp_path / "checkpoints"
    checkpoint_save_path.mkdir()

    with pytest.raises(TypeError):
        iv.models.checkpoint.save_icevision_checkpoint(
            faster_rcnn_model_mmdet_with_num_classes_1,
            str(checkpoint_save_path / "test_checkpoint"),
            meta=[],
        )


def test_model_from_checkpoint(tmp_path):
    model_type = iv.models.mmdet.faster_rcnn
    backbone = model_type.backbones.resnet50_fpn_1x(pretrained=False)
    model = model_type.model(num_classes=1, backbone=backbone)

    checkpoint_save_path = tmp_path / "checkpoints"
    checkpoint_save_path.mkdir()
    iv.models.checkpoint.save_checkpoint(
        model, str(checkpoint_save_path / "test_checkpoint")
    )
    model_loaded = iv.models.checkpoint.model_from_checkpoint(
        str(checkpoint_save_path / "test_checkpoint"),
        model_name="mmdet.faster_rcnn",
        backbone_name="resnet50_fpn_1x",
        classes=["test"],
    )


def test_model_from_checkpoint_for_coco(tmp_path):
    model_type = iv.models.mmdet.faster_rcnn
    backbone = model_type.backbones.resnet50_fpn_1x(pretrained=False)
    model = model_type.model(num_classes=80, backbone=backbone)

    checkpoint_save_path = tmp_path / "checkpoints"
    checkpoint_save_path.mkdir()
    iv.models.checkpoint.save_checkpoint(
        model, str(checkpoint_save_path / "test_checkpoint")
    )
    model_loaded = iv.models.checkpoint.model_from_checkpoint(
        checkpoint_save_path / "test_checkpoint",
        model_name="mmdet.faster_rcnn",
        backbone_name="resnet50_fpn_1x",
        classes=None,
        is_coco=True,
    )
