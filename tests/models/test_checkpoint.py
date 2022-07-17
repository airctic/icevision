import os
import icevision as iv


def test_save_icevision_checkpoint(fridge_faster_rcnn_model_mmdet, tmp_path):
    checkpoint_save_path = tmp_path / "checkpoints"
    checkpoint_save_path.mkdir()
    iv.models.checkpoint.save_checkpoint(
        fridge_faster_rcnn_model_mmdet, str(checkpoint_save_path / "test_checkpoint")
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
