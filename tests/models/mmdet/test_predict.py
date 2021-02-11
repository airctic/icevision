import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "ds, model_type, path, config, weights_path",
    [
        (
            "fridge_ds",
            models.mmdet.faster_rcnn,
            "samples_source",
            "mmdet/configs/faster_rcnn_r50_fpn_1x_coco.py",
            None,
        ),
        (
            "fridge_ds",
            models.mmdet.fcos,
            "samples_source",
            "mmdet/configs/fcos_r50_caffe_fpn_gn-head_1x_coco.py",
            None,
        ),
        (
            "fridge_ds",
            models.mmdet.retinanet,
            "samples_source",
            "mmdet/configs/retinanet_r50_fpn_1x_coco.py",
            None,
        ),
    ],
)
def test_mmdet_bbox_models_predict(ds, model_type, path, config, weights_path, request):
    _, valid_ds = request.getfixturevalue(ds)
    config_path = request.getfixturevalue(path) / config
    model = model_type.model(config_path, num_classes=5, weights_path=weights_path)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=1, shuffle=False)

    batch, _ = first(infer_dl)
    pred = model_type.predict(model, batch)
    assert len(pred) == 1
    assert set(pred[0].keys()) == {"bboxes", "labels", "scores"}

    samps, preds = model_type.predict_dl(model, infer_dl, show_pbar=False)
    assert len(samps) == 1
    assert len(preds) == 1
    assert set(samps[0].keys()) == {
        "bboxes",
        "class_map",
        "filepath",
        "height",
        "imageid",
        "img",
        "labels",
        "width",
    }
    assert samps[0]["img"].shape == (384, 384, 3)
    assert set(preds[0].keys()) == {"bboxes", "labels", "scores"}


def test_mmdet_mask_models_predict(coco_mask_records, samples_source):
    _, valid_records = coco_mask_records[:2], coco_mask_records[:1]

    size = 128
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])
    valid_ds = Dataset(valid_records, valid_tfms)
    model_type = models.mmdet.mask_rcnn

    config_path = samples_source / "mmdet/configs/mask_rcnn_r50_fpn_1x_coco.py"
    model = model_type.model(config_path, num_classes=81)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=1, shuffle=False)
    batch, _ = first(infer_dl)
    pred = model_type.predict(model, batch)
    assert len(pred) == 1
    assert set(pred[0].keys()) == {"bboxes", "labels", "masks", "scores"}

    samps, preds = model_type.predict_dl(model, infer_dl, show_pbar=False)
    assert len(samps) == 1
    assert len(preds) == 1
    assert set(samps[0].keys()) == {
        "areas",
        "bboxes",
        "class_map",
        "filepath",
        "height",
        "imageid",
        "img",
        "iscrowds",
        "labels",
        "masks",
        "width",
    }
    assert samps[0]["img"].shape == (128, 128, 3)
    assert set(preds[0].keys()) == {"bboxes", "labels", "masks", "scores"}
