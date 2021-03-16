import pytest
from icevision.all import *
from copy import deepcopy


@pytest.fixture
def records():
    def _get_record():
        return BaseRecord(
            (
                SizeRecordComponent(),
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

    class_map = ClassMap(["a", "b", "c"])

    record1 = _get_record()
    record1.set_record_id(0)
    record1.set_filepath("none")
    record1.set_img_size(ImgSize(400, 400), original=True)
    record1.detection.set_class_map(class_map)
    record1.detection.add_labels_by_id([2])
    record1.detection.add_bboxes([BBox.from_xywh(10, 10, 200, 200)])

    record2 = _get_record()
    record2.set_record_id(1)
    record2.set_filepath("none")
    record2.set_img_size(ImgSize(500, 500), original=True)
    record2.detection.set_class_map(class_map)
    record2.detection.add_labels_by_id([3, 2])
    record2.detection.add_bboxes(
        [BBox.from_xywh(10, 10, 50, 50), BBox.from_xywh(10, 10, 400, 400)]
    )

    return [record1, record2]


@pytest.fixture()
def preds(records):
    pred = deepcopy(records[0])
    pred.add_component(ScoresRecordComponent())

    pred1 = deepcopy(pred)
    pred1.detection.set_scores([0.9])

    pred2 = deepcopy(pred)
    pred2.detection.set_labels_by_id([3, 2])
    pred2.detection.set_bboxes(
        [BBox.from_xywh(10, 10, 42, 70), BBox.from_xywh(10, 10, 450, 300)]
    )
    pred2.detection.set_scores([0.8, 0.7])

    return [pred1, pred2]


@pytest.fixture()
def expected_coco_output():
    return [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.501",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.300",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.703",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.500",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.500",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.300",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700",
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
