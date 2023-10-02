import pytest
from icevision.all import *
from icevision.models.inference_sahi import IceSahiModel
from sahi.prediction import PredictionResult
import albumentations as A


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
    backbone = model_type.backbones.resnet50_fpn_1x()
    backbone.config_path = samples_source / backbone.config_path

    model = model_type.model(backbone=backbone, num_classes=5)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=1, shuffle=False)
    preds = model_type.predict_from_dl(model, infer_dl, show_pbar=False)
    _test_preds(preds, pred_count=1)


@pytest.mark.parametrize(
    "model_type",
    [models.mmdet.retinanet, models.mmdet.faster_rcnn],
)
def test_sahi(model_type, samples_source, fridge_class_map):
    img_path = samples_source / "sahi/fridge_small_items.jpg"
    tfms_ = tfms.A.Adapter([A.Resize(384, 384), A.Normalize()])

    backbone = model_type.backbones.resnet50_fpn_1x()
    model = model_type.model(backbone=backbone, num_classes=5)

    sahimodel = IceSahiModel(
        model_type=model_type,
        model=model,
        class_map=fridge_class_map,
        tfms=tfms_,
        confidence_threshold=0.4,
    )

    pred = sahimodel.get_sliced_prediction(
        img_path,
        keep_sahi_format=True,
        slice_height=384,
        slice_width=384,
    )
    assert isinstance(pred, PredictionResult)
    assert isinstance(pred.object_prediction_list, list)

    pred = sahimodel.get_sliced_prediction(
        img_path,
        keep_sahi_format=False,
        return_img=True,
        slice_height=384,
        slice_width=384,
    )
    assert isinstance(pred, dict)
    assert sorted(list(pred.keys())) == ["detection", "height", "img", "width"]
    assert sorted(list(pred["detection"].keys())) == [
        "bboxes",
        "label_ids",
        "labels",
        "scores",
    ]


def _test_preds(preds, pred_count=2, mask=False):
    # assert len(preds) == pred_count

    pred = preds[0].pred
    assert isinstance(pred.detection.label_ids, list)
    assert isinstance(pred.detection.bboxes, list)
    assert isinstance(pred.detection.scores, np.ndarray)
    if mask:
        assert isinstance(pred.detection.masks, MaskArray)


@pytest.mark.skip
def test_mmdet_mask_models_predict(coco_mask_records, samples_source):
    valid_records, _ = coco_mask_records[:2], coco_mask_records[:1]

    size = 128
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])
    valid_ds = Dataset(valid_records, valid_tfms)
    model_type = models.mmdet.mask_rcnn

    backbone = model_type.backbones.resnet50_fpn_1x()
    backbone.config_path = samples_source / backbone.config_path

    model = model_type.model(backbone=backbone, num_classes=81)

    infer_dl = model_type.infer_dl(valid_ds, batch_size=2, shuffle=False)
    preds = model_type.predict_from_dl(model, infer_dl, show_pbar=False)
    _test_preds(preds, mask=True)
