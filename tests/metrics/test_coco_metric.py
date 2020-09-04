import pytest
from icevision.all import *
from copy import deepcopy


@pytest.fixture()
def records():
    record1 = {
        "filepath": "none",
        "imageid": 0,
        "labels": [2],
        "bboxes": [BBox.from_xywh(10, 10, 200, 200)],
        "height": 400,
        "width": 400,
    }
    record2 = {
        "filepath": "none",
        "imageid": 1,
        "labels": [3, 2],
        "bboxes": [BBox.from_xywh(10, 10, 50, 50), BBox.from_xywh(10, 10, 400, 400)],
        "height": 500,
        "width": 500,
    }
    return [record1, record2]


@pytest.fixture()
def preds(records):
    pred1 = deepcopy(records[0])
    pred1["scores"] = [0.9]
    pred1.pop("imageid")

    pred2 = {
        "labels": [3, 2],
        "bboxes": [BBox.from_xywh(10, 10, 42, 70), BBox.from_xywh(10, 10, 450, 300)],
        "scores": [0.8, 0.7],
    }
    return [pred1, pred2]


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
    coco_metric.accumulate(records, preds)

    with CaptureStdout() as output:
        coco_metric.finalize()

    assert output == expected_coco_output
