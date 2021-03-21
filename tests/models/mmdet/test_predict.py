import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "ds, model_type",
    [
        (
            "fridge_ds",
            models.mmdet.retinanet,
        ),
    ],
)
def test_mmdet_bbox_models_predict(ds, model_type, samples_source, request):
    _, valid_ds = request.getfixturevalue(ds)
    backbone = model_type.backbones.r50_fpn_1x
    backbone.cfg_url = samples_source / backbone.cfg_url

    model = model_type.model(backbone=backbone, num_classes=5)

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

    backbone = model_type.backbones.r50_fpn_1x
    backbone.cfg_url = samples_source / backbone.cfg_url

    model = model_type.model(backbone=backbone, num_classes=81)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=2, shuffle=False)
    _, records = first(infer_dl)
    pred = model_type.predict(model, [records[0]])
    _test_preds(pred, pred_count=1, mask=True)

    preds = model_type.predict_dl(model, infer_dl, show_pbar=False)
    _test_preds(preds, mask=True)

    assert preds[0].ground_truth.detection.img.shape == (128, 128, 3)
