from icevision.all import *


def _test_preds(preds):
    assert len(preds) == 1

    pred = preds[0]
    assert len(pred["scores"]) == 2

    np.testing.assert_equal(pred["labels"], [2, 3])

    assert len(pred["bboxes"]) == 2
    assert isinstance(pred["bboxes"][0], BBox)


def test_efficient_det_predict(fridge_efficientdet_model, fridge_efficientdet_records):
    fridge_efficientdet_model.eval()

    batch, records = efficientdet.build_infer_batch(fridge_efficientdet_records)
    preds = efficientdet.predict(model=fridge_efficientdet_model, batch=batch)

    _test_preds(preds)


def test_efficient_det_predict_dl(
    fridge_efficientdet_model, fridge_efficientdet_records
):
    fridge_efficientdet_model.eval()

    infer_dl = efficientdet.infer_dl(fridge_efficientdet_records, batch_size=1)
    samples, preds = efficientdet.predict_dl(
        model=fridge_efficientdet_model, infer_dl=infer_dl, show_pbar=False
    )

    _test_preds(preds)


def test_efficient_det_predict_dl_threshold(
    fridge_efficientdet_model, fridge_efficientdet_records
):
    fridge_efficientdet_model.eval()

    infer_dl = efficientdet.infer_dl(fridge_efficientdet_records, batch_size=1)
    samples, preds = efficientdet.predict_dl(
        model=fridge_efficientdet_model,
        infer_dl=infer_dl,
        show_pbar=False,
        detection_threshold=1.0,
    )

    assert len(preds[0]["labels"]) == 0
