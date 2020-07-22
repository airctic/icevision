from mantisshrimp import *
from mantisshrimp.imports import *
from mantisshrimp.models import efficientdet


def test_efficient_det_predict(fridge_efficientdet_model, fridge_efficientdet_records):
    fridge_efficientdet_model.eval()

    batch, records = efficientdet.build_infer_batch(fridge_efficientdet_records)
    preds = efficientdet.predict(model=fridge_efficientdet_model, batch=batch)

    assert len(preds) == 1

    pred = preds[0]
    assert len(pred["scores"]) == 2

    np.testing.assert_equal(pred["labels"], [2, 3])

    assert len(pred["bboxes"]) == 2
    assert isinstance(pred["bboxes"][0], BBox)
