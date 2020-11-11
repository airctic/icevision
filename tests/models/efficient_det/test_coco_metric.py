import pytest
from icevision.all import *


@pytest.fixture
def expected_output():
    return [
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450",
        " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.500",
        " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.500",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.450",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.450",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.450",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.450",
    ]


def test_efficientdet_coco_metric(
    fridge_efficientdet_model, fridge_efficientdet_records, expected_output
):
    fridge_efficientdet_model.eval()

    batch, records = efficientdet.build_valid_batch(fridge_efficientdet_records)

    raw_preds = fridge_efficientdet_model(*batch)

    preds = efficientdet.convert_raw_predictions(raw_preds["detections"], 0)

    coco_metric = COCOMetric(print_summary=True)
    coco_metric.accumulate(records, preds)

    with CaptureStdout() as output:
        coco_metric.finalize()

    assert output == expected_output
