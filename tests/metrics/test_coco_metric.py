import pytest
from icevision.all import *


@pytest.fixture()
def expected_coco_output():
    return [
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.501",
        " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000",
        " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.300",
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.703",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.500",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.500",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.300",
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700",
    ]


def test_coco_eval(records, preds, expected_coco_output):
    coco_eval = create_coco_eval(records, preds, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    with CaptureStdout() as output:
        coco_eval.summarize()

    assert output == expected_coco_output


def test_coco_metric(records, preds, expected_coco_output):
    coco_metric = COCOMetric(print_summary=True)
    preds = [Prediction(pred, gt) for pred, gt in zip(preds, records)]
    coco_metric.accumulate(preds)

    with CaptureStdout() as output:
        coco_metric.finalize()

    assert output == expected_coco_output
