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
    _, records = first(infer_dl)
    pred = model_type.predict(model, records)
    _test_preds(pred, pred_count=1)

    preds = model_type.predict_dl(model, infer_dl, show_pbar=False)
    _test_preds(preds, pred_count=1)

    assert preds[0].ground_truth.detection.img.shape == (384, 384, 3)


def _test_preds(preds, pred_count=2, mask=False):
    # assert len(preds) == pred_count

    pred = preds[0].pred
    assert isinstance(pred.detection.label_ids, list)
    assert isinstance(pred.detection.bboxes, list)
    assert isinstance(pred.detection.scores, np.ndarray)
    if mask:
        assert isinstance(pred.detection.masks, MaskArray)


def test_mmdet_mask_models_predict(coco_mask_records, samples_source):
    valid_records, _ = coco_mask_records[:2], coco_mask_records[:1]

    size = 128
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])
    valid_ds = Dataset(valid_records, valid_tfms)
    model_type = models.mmdet.mask_rcnn

    config_path = samples_source / "mmdet/configs/mask_rcnn_r50_fpn_1x_coco.py"
    model = model_type.model(config_path, num_classes=81)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=2, shuffle=False)
    _, records = first(infer_dl)
    pred = model_type.predict(model, [records[0]])
    _test_preds(pred, pred_count=1, mask=True)

    preds = model_type.predict_dl(model, infer_dl, show_pbar=False)
    _test_preds(preds, mask=True)

    assert preds[0].ground_truth.detection.img.shape == (128, 128, 3)
